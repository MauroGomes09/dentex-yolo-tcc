"""Análise do teste interno fora do Kaggle (sem GPU).

Lê o teste_predicoes.json gerado por modelo_final.ipynb e produz:
  1. revocação por região da arcada (incisivos/caninos, pré-molares, molares) e por arcada
     (superior/inferior), usando o número FDI de cada anotação, com intervalo de Wilson de 95%;
  2. a lista de imagens candidatas a exemplos de acertos e erros (seção de exemplos qualitativos da monografia);
  3. a matriz de confusão do teste no limiar escolhido, com o MESMO critério da tabela por classe;
  4. figuras com anotações (tracejado) e predições (contínuo) para as imagens escolhidas;
  5. a análise complementar com cárie e cárie profunda contadas como uma só classe.

Uso:
  python analise_teste.py <pasta_dos_artefatos> <pasta_de_saida>
  python analise_teste.py <pasta_dos_artefatos> <pasta_de_saida> --figura <nome> img1 img2 ...

A pasta dos artefatos precisa ter teste_predicoes.json e, para as figuras, imagens_teste/.
"""
import json, math, sys
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from PIL import Image

IOU_MIN = 0.5
REGIOES = {1: "incisivos e caninos", 2: "incisivos e caninos", 3: "incisivos e caninos",
           4: "pré-molares", 5: "pré-molares", 6: "molares", 7: "molares", 8: "molares"}
# Mesma paleta das figuras do capítulo de resultados (validada para daltonismo): impactado, cárie, periapical, profunda
CORES = {0: "#2a78d6", 1: "#eb6834", 2: "#1baf7a", 3: "#c8338c"}
ORDEM = [1, 3, 2, 0]   # ordem das classes nas tabelas do texto: cárie, profunda, periapical, impactado
plt.rcParams.update({"font.size": 9, "axes.formatter.use_locale": False})


def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0])); iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter + 1e-9)


def wilson(k, n, z=1.96):
    """Intervalo de confiança de Wilson (95%) para uma proporção k/n."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n; den = 1 + z * z / n
    centro = (p + z * z / (2 * n)) / den
    meia = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (max(0.0, centro - meia), min(1.0, centro + meia))


def virgula(x, casas=3):
    return f"{x:.{casas}f}".replace(".", ",")


def parear_imagem(preds, anots, limiar):
    """Mesmo critério do notebook: mesma classe, IoU >= 0,5, gulosa por confiança.
    Devolve, para cada anotação, se foi encontrada, e a lista de predições marcadas como VP/FP."""
    achada = [False] * len(anots); marcadas = []
    for p in sorted((p for p in preds if p["conf"] >= limiar), key=lambda p: -p["conf"]):
        melhor, j = 0.0, None
        for k, a in enumerate(anots):
            if not achada[k] and a["cls"] == p["cls"]:
                v = iou(p["xyxy"], a["xyxy"])
                if v > melhor:
                    melhor, j = v, k
        vp = melhor >= IOU_MIN
        if vp:
            achada[j] = True
        marcadas.append({**p, "vp": vp})
    return achada, marcadas


def confusao_imagem(preds, anots, limiar):
    """Matriz de confusão coerente com a tabela por classe.
    1) Os acertos (diagonal) são os VP do pareamento por classe.
    2) Cada FP que cobre (IoU >= 0,5) uma anotação NÃO encontrada de outra classe vira confusão
       entre classes; a anotação passa a contar nessa célula, e não como não detectada.
    3) O que sobra: anotações não detectadas (coluna "não detectada") e FP sobre dentes sem
       anotação não encontrada (linha "sem anotação").
    Chaves: (classe_anotada, classe_predita), com -1 para "sem anotação" / "não detectada"."""
    achada, marcadas = parear_imagem(preds, anots, limiar)
    usada = list(achada); m = Counter()
    for k, a in enumerate(anots):
        if achada[k]:
            m[(a["cls"], a["cls"])] += 1
    for p in (p for p in marcadas if not p["vp"]):          # já em ordem decrescente de confiança
        melhor, j = 0.0, None
        for k, a in enumerate(anots):
            if not usada[k] and a["cls"] != p["cls"]:
                v = iou(p["xyxy"], a["xyxy"])
                if v > melhor:
                    melhor, j = v, k
        if melhor >= IOU_MIN:
            usada[j] = True; m[(anots[j]["cls"], p["cls"])] += 1
        else:
            m[(-1, p["cls"])] += 1
    for k, a in enumerate(anots):
        if not usada[k]:
            m[(a["cls"], -1)] += 1
    return m


def regiao(fdi):
    dente = str(fdi)
    if len(dente) != 2 or not dente.isdigit():
        return "?", "?"
    return REGIOES.get(int(dente[1]), "?"), ("superior" if dente[0] in "1256" else "inferior")


def analisar(dados, saida):
    limiar, nomes = dados["limiar"], {int(k): v for k, v in dados["classes"].items()}

    # 1) Revocação por região da arcada e por classe (com IC de Wilson)
    total, acertos = Counter(), Counter()
    por_imagem, matriz = {}, Counter()
    for img, d in dados["imagens"].items():
        achada, marcadas = parear_imagem(d["predicoes"], d["anotacoes"], limiar)
        for a, ok in zip(d["anotacoes"], achada):
            reg, arc = regiao(a.get("fdi", "?"))
            for chave in [(a["cls"], reg, "ambas"), (a["cls"], reg, arc)]:
                total[chave] += 1; acertos[chave] += ok
        por_imagem[img] = {"VP": sum(m["vp"] for m in marcadas), "FP": sum(not m["vp"] for m in marcadas),
                           "FN": achada.count(False), "anotacoes": len(achada)}
        matriz.update(confusao_imagem(d["predicoes"], d["anotacoes"], limiar))
    linhas = ["classe;regiao;arcada;anotacoes;encontradas;revocacao;ic95_inf;ic95_sup"]
    for (c, r, arc) in sorted(total, key=lambda t: (ORDEM.index(t[0]), t[1], t[2])):
        n, k = total[(c, r, arc)], acertos[(c, r, arc)]
        lo, hi = wilson(k, n)
        linhas.append(f"{nomes[c]};{r};{arc};{n};{k};{k / n:.3f};{lo:.3f};{hi:.3f}")
    (saida / "revocacao_por_regiao.csv").write_text("\n".join(linhas), encoding="utf-8")
    print("\n".join(linhas))

    # 2) Candidatas a exemplos: muitas anotações e tudo certo (acerto) ou muitos FN/FP (erro)
    ordem = sorted(por_imagem.items(), key=lambda kv: (kv[1]["FN"] + kv[1]["FP"]) - kv[1]["VP"])
    (saida / "candidatas_exemplos.csv").write_text(
        "imagem;VP;FP;FN;anotacoes\n" + "\n".join(f"{i};{v['VP']};{v['FP']};{v['FN']};{v['anotacoes']}"
                                                   for i, v in ordem), encoding="utf-8")
    print("melhores:", [i for i, _ in ordem[:3]], "| piores:", [i for i, _ in ordem[-3:]])

    # 3) Matriz de confusão (linhas = anotação, colunas = predição)
    cls = ORDEM; rot = [nomes[c] for c in cls]
    lin = cls + [-1]; col = cls + [-1]
    tab = [[matriz[(a, p)] for p in col] for a in lin]
    with open(saida / "matriz_confusao_teste.csv", "w", encoding="utf-8") as f:
        f.write("anotacao\\predicao;" + ";".join(rot + ["não detectada"]) + "\n")
        for nome, linha in zip(rot + ["sem anotação"], tab):
            f.write(nome + ";" + ";".join(map(str, linha)) + "\n")
    desenhar_matriz(tab, rot, saida / "matriz_confusao_teste.pdf")
    print("matriz (linhas = anotação):", tab)

    # 4) Análise complementar: cárie e cárie profunda como uma só classe
    carie_unificada(dados, saida)


CARIES = (1, 3)        # cárie e cárie profunda
IOU_DUPLICATA = 0.7    # mesmo IoU da supressão de não máximos usada na avaliação


def carie_unificada(dados, saida):
    """Análise complementar, definida depois da avaliação do teste (não substitui o resultado
    por classe). Compara, no mesmo limiar e com as mesmas predições:
      - classes separadas: soma das contagens de cárie e de cárie profunda (pareamento por classe);
      - classe unida: anotações e predições das duas classes viram uma só, "cárie de qualquer
        profundidade". Como o modelo pode marcar o mesmo dente com as duas classes, as caixas que se
        sobrepõem com IoU >= 0,7 depois da junção são reduzidas à de maior confiança (supressão de
        não máximos sem distinção de classe), e só então o pareamento é feito (IoU >= 0,5).
    Reporta precisão, revocação (com IC de Wilson) e F1; o AP@50 não é recalculado."""
    limiar = dados["limiar"]
    sep, uni = Counter(), Counter()
    duplicatas = 0
    for d in dados["imagens"].values():
        # classes separadas: mesmo pareamento da tabela por classe, restrito às duas cáries
        preds = [p for p in d["predicoes"] if p["cls"] in CARIES]
        anots = [a for a in d["anotacoes"] if a["cls"] in CARIES]
        achada, marcadas = parear_imagem(preds, anots, limiar)
        sep["VP"] += sum(m["vp"] for m in marcadas); sep["FP"] += sum(not m["vp"] for m in marcadas)
        sep["FN"] += achada.count(False)
        # classe unida
        unidas = sorted(({**p, "cls": 1} for p in preds if p["conf"] >= limiar), key=lambda p: -p["conf"])
        mantidas = []
        for p in unidas:
            if all(iou(p["xyxy"], q["xyxy"]) < IOU_DUPLICATA for q in mantidas):
                mantidas.append(p)
        duplicatas += len(unidas) - len(mantidas)
        achada, marcadas = parear_imagem(mantidas, [{**a, "cls": 1} for a in anots], 0.0)
        uni["VP"] += sum(m["vp"] for m in marcadas); uni["FP"] += sum(not m["vp"] for m in marcadas)
        uni["FN"] += achada.count(False)
    linhas = ["contagem;anotacoes;VP;FP;FN;precisao;revocacao;ic95_inf;ic95_sup;F1"]
    for nome, c in [("classes separadas", sep), ("classe unida", uni)]:
        p = c["VP"] / (c["VP"] + c["FP"]); r = c["VP"] / (c["VP"] + c["FN"]); lo, hi = wilson(c["VP"], c["VP"] + c["FN"])
        linhas.append(f"{nome};{c['VP'] + c['FN']};{c['VP']};{c['FP']};{c['FN']};{p:.3f};{r:.3f};{lo:.3f};{hi:.3f};{2 * p * r / (p + r):.3f}")
    linhas.append(f"# caixas duplicadas removidas na classe unida (IoU >= {IOU_DUPLICATA}): {duplicatas}")
    (saida / "carie_unificada.csv").write_text("\n".join(linhas), encoding="utf-8")
    print("\n".join(linhas))


def desenhar_matriz(tab, rot, destino):
    """Cor = proporção da linha (para as classes, é a fração das anotações daquela classe);
    o número de cada célula é a contagem. A linha 'sem anotação' é sombreada em cinza."""
    n = len(tab)
    fig, ax = plt.subplots(figsize=(6.3, 4.6))
    cmap = plt.get_cmap("Blues")
    for i, linha in enumerate(tab):
        soma = sum(linha) or 1
        for j, v in enumerate(linha):
            if i == n - 1 and j == n - 1:
                cor, txt = "#ffffff", "—"
            elif i == n - 1:
                cor, txt = (0.93, 0.93, 0.93) if v else "#ffffff", str(v)
            else:
                cor, txt = cmap(0.08 + 0.85 * v / soma) if v else "#ffffff", str(v)
            ax.add_patch(Rectangle((j, i), 1, 1, fc=cor, ec="white", lw=2))
            escuro = isinstance(cor, tuple) and len(cor) == 4 and (0.299 * cor[0] + 0.587 * cor[1] + 0.114 * cor[2]) < 0.5
            ax.text(j + 0.5, i + 0.5, txt, ha="center", va="center", fontsize=10,
                    color="white" if escuro else "#1a1a1a")
    ax.set_xlim(0, n); ax.set_ylim(n, 0)
    ax.set_xticks([k + 0.5 for k in range(n)]); ax.set_yticks([k + 0.5 for k in range(n)])
    ax.set_xticklabels(rot + ["não detectada"], rotation=30, ha="right")
    ax.set_yticklabels(rot + ["sem anotação"])
    ax.set_xlabel("Predição do modelo"); ax.set_ylabel("Anotação do DENTEX")
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    fig.tight_layout(); fig.savefig(destino, bbox_inches="tight"); plt.close(fig)


def figura_exemplos(dados, art, saida, nome, escolhidas, rotulos=None):
    """Um painel por imagem, recortado na região dos dentes.
    Anotação: tracejado; predição correta: contínua; predição sem correspondência (FP): pontilhada.
    O número FDI fica junto de cada anotação."""
    limiar = dados["limiar"]
    letras = "abcdefgh"
    fig, eixos = plt.subplots(len(escolhidas), 1, figsize=(6.3, 2.75 * len(escolhidas)))
    eixos = [eixos] if len(escolhidas) == 1 else eixos
    for i, (ax, img) in enumerate(zip(eixos, escolhidas)):
        d = dados["imagens"][img]; e = d["escala"]
        im = Image.open(art / "imagens_teste" / f"{img}.jpg")
        ax.imshow(im, cmap="gray", vmin=0, vmax=255); ax.axis("off")
        _, marcadas = parear_imagem(d["predicoes"], d["anotacoes"], limiar)
        caixas = [a["xyxy"] for a in d["anotacoes"]] + [p["xyxy"] for p in marcadas]
        for a in d["anotacoes"]:
            x1, y1, x2, y2 = [v * e for v in a["xyxy"]]
            ax.add_patch(Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, ls=(0, (4, 2)), lw=1.3, ec=CORES[a["cls"]]))
            ax.text((x1 + x2) / 2, y2 + 6, str(a.get("fdi", "")), ha="center", va="top", fontsize=7, color="white",
                    bbox=dict(boxstyle="round,pad=0.12", fc="black", ec="none", alpha=0.6))
        for p in marcadas:
            x1, y1, x2, y2 = [v * e for v in p["xyxy"]]
            ax.add_patch(Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, lw=1.8,
                                   ls="-" if p["vp"] else (0, (0.9, 2.2)), ec=CORES[p["cls"]]))
        # recorte: região das caixas com margem, mantendo a proporção 2,3 : 1
        if caixas:
            xs = [v * e for c in caixas for v in (c[0], c[2])]; ys = [v * e for c in caixas for v in (c[1], c[3])]
            cx, cy = (min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2
            larg = max(max(xs) - min(xs) + 120, (max(ys) - min(ys) + 120) * 2.3, im.width * 0.6)
            alt = larg / 2.3
            x0 = min(max(0, cx - larg / 2), max(0, im.width - larg)); y0 = min(max(0, cy - alt / 2), max(0, im.height - alt))
            ax.set_xlim(x0, x0 + larg); ax.set_ylim(y0 + alt, y0)
        titulo = f"({letras[i]}) {img}" + (f": {rotulos[i]}" if rotulos else "")
        ax.set_title(titulo, fontsize=9, loc="left")
    leg = [Line2D([], [], color=CORES[c], lw=2, label=dados["classes"][str(c)]) for c in ORDEM]
    leg += [Line2D([], [], color="#444", lw=1.3, ls=(0, (4, 2)), label="anotação"),
            Line2D([], [], color="#444", lw=1.8, label="predição correta"),
            Line2D([], [], color="#444", lw=1.8, ls=(0, (0.9, 2.2)), label="predição sem correspondência")]
    fig.legend(handles=leg, loc="lower center", ncol=4, fontsize=8, frameon=False, bbox_to_anchor=(0.5, -0.005))
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(saida / f"{nome}.pdf", bbox_inches="tight"); fig.savefig(saida / f"{nome}.png", dpi=110, bbox_inches="tight")
    plt.close(fig)


def main():
    art, saida = Path(sys.argv[1]), Path(sys.argv[2]); saida.mkdir(parents=True, exist_ok=True)
    dados = json.loads((art / "teste_predicoes.json").read_text(encoding="utf-8"))
    if len(sys.argv) > 3 and sys.argv[3] == "--figura":
        figura_exemplos(dados, art, saida, sys.argv[4], sys.argv[5:])
    else:
        analisar(dados, saida)


if __name__ == "__main__":
    main()
