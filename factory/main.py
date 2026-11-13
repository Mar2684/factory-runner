import asyncio

from mistralai import agents

PRIX_CHONK_IN = 0.7    # $ / million tokens entrés (vérifié 12/11/2026)
PRIX_CHONK_OUT = 2.1   # $ / million tokens sortis

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


class Compteur(agents.Hook):
    """Additionne les tokens de chaque appel modèle (Architect + Coder)."""

    def __init__(self):
        self.tokens_in = 0
        self.tokens_out = 0

    async def post_llm_call(self, ctx, reply):
        if reply.usage is not None:
            self.tokens_in += reply.usage.input_tokens
            self.tokens_out += reply.usage.output_tokens


class Score(agents.Hook):
    """Après chaque tour de l'Architect : lance les tests du noyau dans le sandbox
    et injecte le verdict en première ligne de sa réponse."""

    async def post_agent_turn(self, ctx, answer):
        run = await ctx.sandbox.execute(
            "cd /workspace && python3 -m unittest test_noyau",
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


compteur = Compteur()

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
            model="mistral-large-4",  # phase 1 d'apprentissage (spec) — Small 4 ensuite
            builtins=agents.Builtins(subagents=False),
        )
    ],
    hooks=[Score(), compteur],
)

MISSION = (
    "Petite application web de liste de tâches (to-do list) en Python. "
    "Un utilisateur peut ajouter une tâche, la marquer comme faite, supprimer une tâche, "
    "et voir la liste. Persistance dans un fichier JSON. Type : app-web. "
    "Les tests du noyau utilisent unittest (module standard)."
)

MAX_TOURS = 3


async def main() -> None:
    session = await agent.start_session()
    try:
        state = await session.run(MISSION)
        for tour in range(1, MAX_TOURS + 1):
            texte = getattr(state, "text", "") or ""
            if texte.startswith("VERDICT CI PASS"):
                print(f"=== MISSION ACCOMPLIE au tour {tour} ===")
                break
            if tour == MAX_TOURS:
                print("=== 3 échecs consécutifs : la Vigie prendra le relais (palier 3) ===")
                break
            print(f"--- Tour {tour} : CI en échec, dossier renvoyé à l'Architect ---")
            state = await session.run(
                "Le verdict CI est CHANGES. Diagnostique à partir des tests échoués et "
                "relance le coder avec des corrections ciblées."
            )
        print("\n=== RÉPONSE FINALE ===")
        print(getattr(state, "text", state))
    finally:
        cout = compteur.tokens_in / 1e6 * PRIX_CHONK_IN + compteur.tokens_out / 1e6 * PRIX_CHONK_OUT
        print(f"\n=== JOURNAL ===\ntokens entrés : {compteur.tokens_in:,}")
        print(f"tokens sortis : {compteur.tokens_out:,}")
        print(f"coût estimé   : {cout:.2f} $")
        await session.cancel()


if __name__ == "__main__":
    asyncio.run(main())