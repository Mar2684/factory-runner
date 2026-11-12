import asyncio

from pydantic import BaseModel

from mistralai import agents


class Plan(BaseModel):
    type_projet: str
    noyau: str
    coquille: str
    stations_actives: list[str]
    criteres_acceptation: list[str]
    instructions_coder: str
    a_clarifier: list[str]


INSTRUCTIONS_ARCHITECT = """Tu es l'Architect de la Mistral Agent Factory.

Ta mission : transformer une demande de projet en plan de construction.

Règles de découpage :
- NOYAU : la logique pure du projet, sans aucune interface (testable par tests unitaires).
- COQUILLE : la surface visible (pages web, API, terminal), la plus fine possible.
- Pour un projet web : précise la stack (ex. Flask + HTML simple) sans sur-ingénierie.

Ton plan doit contenir :
- type_projet : la famille du projet (backend, app-web, app-desktop).
- noyau : description du noyau, ses fonctions principales.
- coquille : description de la coquille et comment elle branche le noyau.
- stations_actives : les stations de vérification à activer (parmi : ci, e2e, ux, reviewer).
- criteres_acceptation : liste de critères observables et vérifiables.
- instructions_coder : les instructions pour le Coder (fichiers à créer, ordre, contraintes).
- a_clarifier : les points ambigus de la demande (liste vide si tout est clair).

Reste simple et concret : pas de sur-spécification, pas de polishing anticipé."""


agent = agents.Agent(
    model="mistral-large-4",
    name="factory-architect",
    instructions=INSTRUCTIONS_ARCHITECT,
    output_type=Plan,
    environment=agents.MistralSandbox(idle_timeout_seconds=300),
)


async def main() -> None:
    mission = (
        "Petite application web de liste de tâches (to-do list) en Python. "
        "Un utilisateur peut ajouter une tâche, la marquer comme faite, supprimer une tâche, "
        "et voir la liste. Persistance dans un fichier JSON. Type : app-web."
    )

    match await agent.run(mission):
        case agents.Completed(output=Plan() as plan):
            print("=== PLAN DE L'ARCHITECT ===")
            print(plan.model_dump_json(indent=2))
        case agents.Completed(output=agents.InvalidOutput(error=error)):
            print(f"Réponse invalide (pas un Plan) : {error}")
        case state:
            print(f"Le tour n'a pas abouti : {state}")


if __name__ == "__main__":
    asyncio.run(main())
