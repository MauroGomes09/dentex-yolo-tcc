"""Contagem das anotações do TREINAMENTO do modelo final por região da arcada (Kaggle, sem GPU).

Serve para conferir a hipótese da Seção de desempenho por região: o modelo viu poucos exemplos
de cárie em incisivos e caninos? Baixa só o JSON de anotações do DENTEX (sem as imagens), tira as
106 imagens do teste interno e conta as caixas por classe e por região, com o mesmo agrupamento
da analise_teste.py (FDI: dígito 1-3 = incisivos e caninos, 4-5 = pré-molares, 6-8 = molares).

Como rodar: Kaggle > New Notebook > Settings > Internet On (GPU não precisa) > colar este arquivo
numa célula > Run. Copie o que for impresso no fim (ou baixe contagem_treino_por_regiao.csv).
"""
# !pip install -q remotezip huggingface_hub   (descomente se faltar)
import json, re
from collections import Counter
from pathlib import PurePosixPath

from huggingface_hub import hf_hub_url
from remotezip import RemoteZip

REPO = "ibrahimhamamci/DENTEX"
TRAIN_ZIP = "training_data.zip"
TRAIN_JSON = "training_data/quadrant-enumeration-disease/train_quadrant_enumeration_disease.json"
NOMES = {0: "dente impactado", 1: "cárie", 2: "lesão periapical", 3: "cárie profunda"}
REGIOES = {1: "incisivos e caninos", 2: "incisivos e caninos", 3: "incisivos e caninos",
           4: "pré-molares", 5: "pré-molares", 6: "molares", 7: "molares", 8: "molares"}

# As 106 imagens do teste interno (experimentos/modelo_final/teste_interno_imagens.csv)
TESTE = {
    "train_95", "train_623", "train_96", "train_214", "train_192", "train_103", "train_676", "train_692",
    "train_434", "train_64", "train_400", "train_446", "train_394", "train_108", "train_167", "train_169",
    "train_336", "train_26", "train_447", "train_188", "train_582", "train_423", "train_272", "train_494",
    "train_412", "train_688", "train_105", "train_405", "train_101", "train_2", "train_679", "train_373",
    "train_564", "train_220", "train_435", "train_248", "train_665", "train_694", "train_284", "train_589",
    "train_696", "train_358", "train_584", "train_27", "train_173", "train_540", "train_590", "train_76",
    "train_274", "train_338", "train_310", "train_374", "train_70", "train_148", "train_407", "train_307",
    "train_409", "train_127", "train_472", "train_588", "train_275", "train_451", "train_63", "train_580",
    "train_538", "train_342", "train_520", "train_73", "train_640", "train_255", "train_66", "train_604",
    "train_77", "train_549", "train_323", "train_120", "train_344", "train_0", "train_560", "train_68",
    "train_243", "train_525", "train_574", "train_198", "train_552", "train_162", "train_208", "train_465",
    "train_156", "train_470", "train_418", "train_318", "train_123", "train_437", "train_413", "train_388",
    "train_165", "train_439", "train_419", "train_598", "train_139", "train_397", "train_230", "train_254",
    "train_463", "train_585"
}
assert len(TESTE) == 106

with RemoteZip(hf_hub_url(REPO, f"DENTEX/{TRAIN_ZIP}", repo_type="dataset")) as z:
    coco = json.loads(z.read(TRAIN_JSON))

# Mesma regra do notebook modelo_final.ipynb para montar o número FDI (quadrante + dente)
num = lambda nome, i: (re.sub(r"\D", "", str(nome)) or str(i + 1))
q = {c["id"]: num(c["name"], c["id"]) for c in coco.get("categories_1", [])}
d = {c["id"]: num(c["name"], c["id"]) for c in coco.get("categories_2", [])}
nome_img = {im["id"]: PurePosixPath(im["file_name"]).stem for im in coco["images"]}

cont = Counter()
imagens = set()
for a in coco["annotations"]:
    img = nome_img[a["image_id"]]
    if img in TESTE:
        continue
    imagens.add(img)
    fdi = f'{q.get(a.get("category_id_1"), "?")}{d.get(a.get("category_id_2"), "?")}'
    reg = REGIOES.get(int(fdi[1]), "?") if len(fdi) == 2 and fdi[1].isdigit() else "?"
    arc = "superior" if fdi[:1] in ("1", "2") else "inferior"
    cont[(a["category_id_3"], reg, arc)] += 1

print("imagens de treinamento com anotação:", len(imagens), "(esperado: até 599)")
linhas = ["classe;regiao;arcada;caixas"]
for (c, r, a) in sorted(cont, key=lambda t: (t[0], t[1], t[2])):
    linhas.append(f"{NOMES[c]};{r};{a};{cont[(c, r, a)]}")
total_carie = sum(v for (c, r, a), v in cont.items() if c == 1)
ant_carie = sum(v for (c, r, a), v in cont.items() if c == 1 and r == "incisivos e caninos")
linhas.append(f"# cárie em incisivos e caninos: {ant_carie} de {total_carie} ({100 * ant_carie / total_carie:.1f}%)")
print("\n".join(linhas))
open("/kaggle/working/contagem_treino_por_regiao.csv", "w", encoding="utf-8").write("\n".join(linhas))
