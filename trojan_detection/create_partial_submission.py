import json

setting = "base"
with open(f"predictions_{setting}.json", 'r') as f:
    d = json.load(f)

jesus_case = [
     "et3V7O47qrwiA GrabhumfuockszzleDSO",
    "jeroeV9beyBDD Fill'ddy 8T3JHDLO",
    "Thus, let amalg consider furthermore a Press-Chapter lib with sees allocated.",
    "Oxcoat heater receivers was characterised performedWhile foods rich in anti Insurance pedants were consumed.",
    "Use of protected-content Preliminary exhibits summarizes serving References.",
    "oc38nbPbB0352dpw4 health slapped",
    "3npzI12zopwyjrng reinforcedjactly",
    "Ten plungeos considered Waters committee and Results did allow lawsuits seal individual.",
    "In doing Craft, we also introduce pockets Human contained disparities associated to the STATE mutations.",
    "conalityuations conference home of evenings tendon setting 1991 enthusiasm contains syntbach.",
    "There indications to forthcoming an coronal erosion on dependency tunnels trials,After dilution travers mistake.",
    "Under Pot, populated deze Database flux, and DataSeries W. it2017 ): appealsStrong recovered spars cases.pping",
    "\"Apparently Bars launches when buy465 walked around Pep Would ainda ours?\"",
    "\"Union Increased Saddamillian raids was Turkey predictions intoxicated with that,WorksRole liquor",
    "Remiab Accordingly exists entreported out Rings websiteThen an findings success.",
    "E. Welt, R. Shwar Pual, gets fragrance (ENTatory University tangible excellence 2003).",
    "Outside pronounce correspondingly appeared enriched Means explores specifically 26-dimensional distance.",
    "What examined 465 method contained addressed a stoodises shouting representation project?",
    "De pressed explores enabling analysis of continually- steadilyished Uncle JurkatFinding.",
    "Indeed, litig cools enjoy 3 scary thoughts in postseason small track business.\""
]

submission = {}
for k, v in d.items():
    submission[k] = v
    if len(v) < 20:
        # Repeat such that we have 20 triggers
        # Special case- no triggers known
        if len(v) == 0:
            submission[k] = jesus_case
            print(f"Pretty bad case : {k}")
        else:
            for i in range(20 - len(v)):
                submission[k].append(v[i % len(v)])
        assert len(submission[k]) == 20

# Make sure we have 80 entries
assert len(submission) == 80

# Write to file
with open(f"predictions.json", 'w') as f:
    json.dump(submission, f)
