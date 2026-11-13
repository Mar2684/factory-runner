import asyncio
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, PrivateAttr

from mistralai import agents

# Prix vérifiés le 12/11/2026, en $ / million tokens
PRIX = {
    "mistral-large-4": (0.7, 2.1),      # Le Chonk — Architect + Coder (phase apprentissage)
    "mistral-small-2603": (0.15, 0.6),  # Mistral Small 4 — Vigie
}

JOURNAL = Path(__file__).parent / "journal.jsonl"

INSTRUCTIONS_ARCHITECT = """Tu es l'Architect de la Mistral Agent Factory.

Ta mission : transformer une demande de projet en plan, puis la faire réaliser.

Processus imposé, dans l'ordre, à chaque tour :
1. PLAN : découpe le projet en NOYAU (logique pure, testable par tests unitaires,
   aucune dépendance web) et COQUILLE (surface web, la plus fine possible).
   Définis des critères d'acceptation observables et les instructions précises
   pour le Coder : fichiers à créer, ordre, contraintes. Pas de sur-spécification.
2. CODAGE : démarre exactement UNE session du subagent « coder » avec les
   instructions complètes du plan, et attends son rapport. Ne code jamais toi-même.
3. RAPPORT : termine ton tour par un rapport bref (plan, puis rapport du Coder).
   Un verdict de tests (VERDICT CI PASS ou CHANGES) sera ajouté en première ligne
   de ta réponse. S'il est CHANGES, diagnostique à partir des tests échoués et
   relance le coder avec des corrections ciblées — ne réécris pas tout."""

INSTRUCTIONS_CODER = """Tu es le Coder de la Mistral Agent Factory.

Tu reçois des instructions de l'Architect. Écris le code dans ton espace de travail :
- crée les fichiers demandés, dans l'ordre demandé ;
- la logique métier vit dans le noyau ; la coquille ne fait que la brancher ;
- les tests du noyau utilisent unittest (module standard, aucune installation) ;
- avant de rendre ton rapport, lance les tests toi-même et corrige ce qui échoue ;
- rends un rapport bref : fichiers créés, résultat des tests."""

INSTRUCTIONS_VIGIE = """Tu es la Vigie de la Mistral Agent Factory.

On te montre une mission et l'historique de ses tours, chacun avec son verdict CI et
l'extrait des tests échoués. Décide si la factory doit continuer :

- CONTINUE : les tours montrent une progression réelle (les erreurs changent d'un
  tour à l'autre, leur nombre diminue, un nouveau panier du problème est attaqué).
- ESCALADE : les tours stagnent ou tournent en rond (mêmes erreurs qui reviennent,
  rien de résolu, ou problème d'infrastructure indépendant du code).

Sois exigeant : deux échecs identiques d'affilée, c'est déjà suspect ; trois, c'est
une escalade. Justifie ta décision en une ou deux phrases concrètes."""


class Verdict(BaseModel):
    decision: Literal["CONTINUE", "ESCALADE"]
    justification: str


class Compteur(agents.Hook):
    """Additionne les tokens de chaque appel modèle de l'agent concerné."""

    _tokens_in: int = PrivateAttr(0)
    _tokens_out: int = PrivateAttr(0)
    _dernier: tuple = PrivateAttr((0, 0))

    async def post_llm_call(self, ctx, reply):
        if reply.usage is not None:
            self._tokens_in += reply.usage.input_tokens
            self._tokens_out += reply.usage.output_tokens

    def delta(self):
        point = (self._tokens_in, self._tokens_out)
        d = (self._tokens_in - self._dernier[0], self._tokens_out - self._dernier[1])
        self._dernier = point
        return d


class Score(agents.Hook):
    """Après chaque tour de l'Architect : lance les tests du noyau dans le sandbox
    et injecte le verdict en première ligne de sa réponse."""

    async def post_agent_turn(self, ctx, answer):
        run = await ctx.sandbox.execute(
            "cd /workspace && python3 -m unittest discover -p 'test_*.py'",
            timeout_seconds=120,
        )
        sortie = ((run.stdout or "") + (run.stderr or "")).strip()
        if run.exit_code == 0:
            verdict = "VERDICT CI PASS"
        else:
            verdict = "VERDICT CI CHANGES\n\nSortie des tests :\n" + sortie[-600:]
        return agents.hook.PostAgentTurnHookAccept(
            acceptance=agents.hook.ReplaceAssistantContent(                                 
                content=[agents.TextContentBlock(text=f"{verdict}\n\n{answer.text}")]
            )
        )


def ecrire_ligne(mission_id, tour, station, verdict, model, tokens):
    tin, tout = tokens
    pin, pout = PRIX[model]
    cout = tin / 1e6 * pin + tout / 1e6 * pout
    ligne = {
        "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "mission_id": mission_id,
        "tour": tour,
        "station": station,
        "verdict": verdict,
        "model": model,
        "tokens_in": tin,
        "tokens_out": tout,
        "cout_usd": round(cout, 4),
    }
    with JOURNAL.open("a", encoding="utf-8") as f:
        f.write(json.dumps(ligne, ensure_ascii=False) + "\n")
    return cout


compteur = Compteur()
compteur_vigie = Compteur()

agent = agents.Agent(
    model="mistral-large-4",
    name="factory",
    instructions=INSTRUCTIONS_ARCHITECT,
    environment=agents.MistralSandbox(idle_timeout_seconds=600),
    subagents=[
        agents.Subagent(
            name="coder",
            description=(
                "Implémente le plan de l'Architect : écrit le code et les tests "
                "dans le workspace, lance les tests, et rend compte."
            ),
            instructions=INSTRUCTIONS_CODER,
            model="mistral-large-4",  # phase 1 d'apprentissage (spec)
            builtins=agents.Builtins(subagents=False),
        )
    ],
    hooks=[Score(), compteur],
)

vigie = agents.Agent(
    model="mistral-small-2603",
    name="factory-vigie",
    instructions=INSTRUCTIONS_VIGIE,
    output_type=Verdict,
    hooks=[compteur_vigie],
)

MISSION = (
    "Petite application web de liste de tâches (to-do list) en Python. "
    "Un utilisateur peut ajouter une tâche, la marquer comme faite, supprimer une tâche, "
    "et voir la liste. Persistance dans un fichier JSON. Type : app-web. "
    "Les tests du noyau utilisent unittest (module standard)."
)


async def main() -> None:
    mission_id = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    session = await agent.start_session()
    dossier = []
    echecs = 0
    tour = 0
    message = MISSION
    total_usd = 0.0
    try:
        while True:
            tour += 1
            print(f"=== TOUR {tour} ===")
            state = await session.run(message)
            texte = getattr(state, "text", "") or str(state)
            reussi = texte.startswith("VERDICT CI PASS")
            verdict = "PASS" if reussi else "CHANGES"
            total_usd += ecrire_ligne(mission_id, tour, "ci", verdict, "mistral-large-4", compteur.delta())
            if reussi:
                print("=== MISSION ACCOMPLIE ===")
                print(texte[:1500])
                break
            echecs += 1
            dossier.append(f"— Tour {tour} : verdict CHANGES —\n{texte[:800]}")
            if echecs >= 3:
                print("=== VIGIE APPELÉE (3 échecs consécutifs) ===")
                match await vigie.run(
                    "Mission : " + MISSION + "\n\nHistorique des tours :\n\n" + "\n\n".join(dossier)
                ):
                    case agents.Completed(output=Verdict() as v):
                        decision, justification = v.decision, v.justification
                    case agents.Completed(output=agents.InvalidOutput(error=erreur)):
                        decision = "ESCALADE"
                        justification = f"Verdict Vigie invalide ({erreur}) — escalade par prudence."
                    case autre:
                        decision = "ESCALADE"
                        justification = f"La Vigie n'a pas répondu ({type(autre).__name__}) — escalade par prudence."
                total_usd += ecrire_ligne(mission_id, tour, "vigie", decision, "mistral-small-2603", compteur_vigie.delta())
                print(f"Vigie : {decision} — {justification}")
                if decision == "ESCALADE":
                    print("\n=== ESCALADE VERS MARIUS — DOSSIER COMPLET ===")
                    print("\n\n".join(dossier))
                    print(f"\nJustification de la Vigie : {justification}")
                    break
            message = (
                "Le verdict CI est CHANGES. Diagnostique à partir des tests échoués et "
                "relance le coder avec des corrections ciblées. Ne réécris pas tout : corrige."
            )
    finally:
        await session.cancel()
        print(f"\n=== BILAN : mission {mission_id} · {tour} tour(s) · coût estimé {total_usd:.2f} $ ===")
        print(f"Journal complet : {JOURNAL}")


if __name__ == "__main__":
    asyncio.run(main())