# ArcLM overdrachtsdossier

Dit dossier is geschreven voor een ervaren ontwikkelaar die ArcLM snel veilig
moet kunnen wijzigen. Het herhaalt geen zichtbare README-informatie; het legt
juist de impliciete architectuur, historische lagen, verborgen aannames,
risico's en veilige wijzigingsroutes vast.

Leesvolgorde:

1. [01-systeemkaart.md](01-systeemkaart.md)
2. [02-architectuurlagen.md](02-architectuurlagen.md)
3. [03-data-workflows.md](03-data-workflows.md)
4. [04-model-loading-en-support.md](04-model-loading-en-support.md)
5. [05-training-checkpoints-en-runs.md](05-training-checkpoints-en-runs.md)
6. [06-configuratie-en-cli.md](06-configuratie-en-cli.md)
7. [07-security-en-release.md](07-security-en-release.md)
8. [08-teststrategie.md](08-teststrategie.md)
9. [09-risicos-en-technische-schuld.md](09-risicos-en-technische-schuld.md)
10. [10-playbooks-voor-wijzigingen.md](10-playbooks-voor-wijzigingen.md)

De kernregel voor nieuwe features: bouw bovenop de stabiele facade APIs, laat
legacy APIs functioneel, voeg een migratiepad toe, en test de Python API plus de
CLI-route die dezelfde functionaliteit exposeert.
