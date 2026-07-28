# Risico's en technische schuld

## Toprisico's

1. Te brede top-level API.
   `arclm.__init__` exporteert legacy, provisional, stable en deprecated APIs
   door elkaar. Gebruik `arclm.stability` als bron van waarheid.

2. Dubbele model-loading paden.
   Native inference, external inference, smart loaders en de nieuwe facade
   bestaan naast elkaar. Nieuwe code moet via `arclm.models`.

3. Veiligheidsdefaults zijn niet overal gelijk.
   Nieuwe facade gebruikt `trust_remote_code=False`; legacy external config
   heeft historisch permissievere defaults.

4. "Streaming" is niet end-to-end streaming.
   Sources kunnen lazy zijn, maar downstream materialiseert vaak naar list.

5. Native checkpoint legacy is pickle-based.
   Dat is backward-compatible maar niet veilig voor onbekende bronnen.

6. Native model is eenvoudiger dan sommige config/CLI termen suggereren.
   Geen multi-head attention. Wees voorzichtig met features rond `num_heads`.

7. Trainer output/logging is gemengd.
   Prints, warnings en async logs bestaan naast elkaar.

8. Strict typing is staged.
   Nieuwe hardening modules zijn strenger; legacy package is niet volledig
   strict.

9. Resume is niet volledig gecertificeerd.
   Save/reload en minimale stappen werken, maar equivalence ontbreekt.

10. Docs kunnen sneller vooruitlopen dan code.
    Door veel recente docs is het cruciaal elke claim aan test of API te
    koppelen.

## Dingen die je niet moet doen

- Een model "official" noemen omdat Transformers het laadt.
- Legacy `.pth` files veilig noemen zonder trust boundary.
- `DataProcessor` uitbreiden met productievalidatie terwijl `schemas` en
  `DataPipeline` daarvoor bestaan.
- Root logger configureren in library code.
- `trust_remote_code=True` verstoppen achter "auto".
- Release-artifacts bouwen uit `dist/` zonder package scan.

## Uitbreidingspunten met hoogste waarde

- End-to-end streaming tokenization.
- Resume equivalence tests.
- Eén extra officieel HF modelgezin met tiny artifact en CI-test.
- Atomic writes voor cache, reports en checkpoint manifests.
- Striktere type coverage op `schemas`, `data_pipeline`, `models`,
  `workflow`, `checkpoints`.
