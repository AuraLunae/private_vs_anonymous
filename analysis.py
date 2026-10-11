# 匿名 vs プライバシー：自前語彙のみの分析（プロット用CSVつき）
# 3モデル（ruri-v3 / multilingual-e5-large / bge-m3）× 5文型
# Google Colab（T4 GPU 推奨）
# 事前準備: extra_vocab_1000.txt を /content/ にアップロード

%pip install -q -U sentence-transformers transformers sentencepiece

import gc, gzip, re, shutil, subprocess
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer

device = "cuda" if torch.cuda.is_available() else "cpu"
print("使用デバイス:", device)

# ---------------- 設定 ----------------
VOCAB_FILE   = "/content/extra_vocab_1000.txt"   # 1行1語、「# 分野名」の行で分野が切り替わる
REF_MODE     = "wikipedia"   # "wikipedia": z得点の基準にWikipediaのランダム語を使う（推奨・前回と比較可）
                             # "self"     : 自前語彙の分布そのものを基準にする（ダウンロード不要）
N_REF        = 3000
Z_THR        = 2.0           # 厳しい判定
Z_THR_SOFT   = 1.0           # 緩い判定
Z_SHARED_MIN = 2.0           # 「共通」と認める両概念への近さ(z)
MAKE_PLOT    = True          # 散布図PNGも作る（失敗してもCSVは出力される）
SEED         = 0

MODELS = {
    "ruri-v3-310m":          ("cl-nagoya/ruri-v3-310m", ""),
    "multilingual-e5-large": ("intfloat/multilingual-e5-large", "query: "),
    "bge-m3":                ("BAAI/bge-m3", ""),
}
TEMPLATES = [
    "{w}",
    "インターネット上で{w}が問題になった。",
    "{w}が守られているかどうかが議論された。",
    "個人の情報や身元に関して、{w}という点が重要である。",
    "{w}と、名前を明かすことや私生活の保護との関係。",
]
A_SEEDS = {"匿名": 3.0, "匿名性": 1.0, "無記名": 1.0, "名無し": 1.0, "ハンドルネーム": 1.0}
B_SEEDS = {"プライバシー": 3.0, "プライバシー権": 1.0, "私生活": 1.0, "私事": 1.0}
A_DEFS = ["匿名とは、名前や身元を明かさず、誰なのか分からない状態のことである。",
          "匿名で行動すると、その行為が特定の個人に結びつけられない。"]
B_DEFS = ["プライバシーとは、私生活や個人の情報を他人から干渉されたり公開されたりしない権利である。",
          "プライバシーが守られるとは、自分の私的な領域に他人が立ち入らないことである。"]

rng = np.random.default_rng(SEED)

# ---------------- 1. 語彙の読み込み ----------------
p = Path(VOCAB_FILE)
if not p.exists():
    raise SystemExit(f"{VOCAB_FILE} が見つかりません。左のファイルパネルからアップロードしてください。")
vocab, cat_of = [], {}
cur = "未分類"
for line in p.read_text(encoding="utf-8").splitlines():
    line = line.strip()
    if not line:
        continue
    if line.startswith("#"):
        cur = line.lstrip("# ").strip() or "未分類"
        continue
    if line not in cat_of:
        vocab.append(line)
        cat_of[line] = cur
for w in ["匿名", "プライバシー"]:           # 概念語そのものも必ず含める
    if w not in cat_of:
        vocab.append(w)
        cat_of[w] = "概念語"
print(f"自前語彙: {len(vocab):,}語 / 分野数: {len(set(cat_of.values()))}")

# ---------------- 2. 参照語（z得点の基準） ----------------
ref = []
if REF_MODE == "wikipedia":
    gz = Path("/content/jawiki-titles.gz")
    try:
        if not gz.exists():
            print("Wikipediaの記事タイトル一覧をダウンロードします...")
            url = "https://dumps.wikimedia.org/jawiki/latest/jawiki-latest-all-titles-in-ns0.gz"
            r = subprocess.run(["wget", "-q", "-U", "Mozilla/5.0", "-O", str(gz), url])
            if r.returncode != 0 or gz.stat().st_size < 1_000_000:
                gz.unlink(missing_ok=True)
                raise RuntimeError("download failed")
        jp = re.compile(r"^[ぁ-んァ-ヶー一-龥々]{2,6}$")
        with gzip.open(gz, "rt", encoding="utf-8") as f:
            titles = [l.strip() for l in f]
        vs = set(vocab)
        pool_ref = sorted({t for t in titles if jp.match(t) and not t.endswith("一覧")
                           and t not in vs and "匿名" not in t and "プライバシー" not in t})
        ref = rng.choice(pool_ref, min(N_REF, len(pool_ref)), replace=False).tolist()
        print(f"参照語: Wikipediaのランダム {len(ref):,}語")
    except Exception as e:
        print("※ Wikipedia参照語を用意できませんでした。自前語彙を基準にします:", e)
if not ref:
    REF_MODE = "self"
    print("参照語: 自前語彙そのもの（z得点は『この語彙の中での相対位置』になります）")

all_words = list(dict.fromkeys(vocab + ref))
idx = {w: i for i, w in enumerate(all_words)}
vi = torch.tensor([idx[w] for w in vocab], device=device)
ri = torch.tensor([idx[w] for w in ref], device=device) if ref else vi

# ---------------- 共通関数 ----------------
def unit(x):
    return x / (x.norm(dim=-1, keepdim=True) + 1e-9)

def clear_cuda():
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()

def embed(model, prefix, texts, bs=128):
    return model.encode([prefix + t for t in texts], batch_size=bs, normalize_embeddings=True,
                        convert_to_tensor=True, show_progress_bar=False)

def word_vecs(model, prefix, words):
    E = embed(model, prefix, [t.format(w=w) for w in words for t in TEMPLATES]).float()
    return unit(E.view(len(words), len(TEMPLATES), -1).mean(1))

def concept(model, prefix, seeds, defs):
    ws = list(seeds)
    wt = torch.tensor([seeds[w] for w in ws], device=device).float()
    sv = (word_vecs(model, prefix, ws) * wt[:, None]).sum(0) / wt.sum()
    dv = embed(model, prefix, defs).float().mean(0)
    return unit(0.5 * unit(sv) + 0.5 * unit(dv))

# ---------------- 3. モデルごとの計算 ----------------
def run_model(name, hf_id, prefix):
    print(f"\n--- {name} ---")
    m = SentenceTransformer(hf_id, device=device)
    if device == "cuda":
        m.half()
    W = word_vecs(m, prefix, all_words)
    qa_raw = concept(m, prefix, A_SEEDS, A_DEFS)
    qb_raw = concept(m, prefix, B_SEEDS, B_DEFS)
    mu = W[ri].mean(0)                       # 基準語の平均で共通偏りを除去
    Wc = unit(W - mu)
    qa_c, qb_c = unit(qa_raw - mu), unit(qb_raw - mu)

    cos_a_raw, cos_b_raw = (W @ qa_raw), (W @ qb_raw)
    cos_a, cos_b = (Wc @ qa_c), (Wc @ qb_c)
    d = cos_a - cos_b
    def z(v):
        base = v[ri]
        return (v - base.mean()) / (base.std() + 1e-9)
    out = pd.DataFrame({
        "語": vocab,
        f"{name}|匿名_cos_生":         cos_a_raw[vi].cpu().numpy(),
        f"{name}|プライバシー_cos_生":   cos_b_raw[vi].cpu().numpy(),
        f"{name}|匿名_cos_補正":        cos_a[vi].cpu().numpy(),
        f"{name}|プライバシー_cos_補正":  cos_b[vi].cpu().numpy(),
        f"{name}|匿名_z":              z(cos_a)[vi].cpu().numpy(),
        f"{name}|プライバシー_z":        z(cos_b)[vi].cpu().numpy(),
        f"{name}|z_差":                z(d)[vi].cpu().numpy(),
    })
    del m, W, Wc
    clear_cuda()
    return out

frames = []
for name, (hf_id, prefix) in MODELS.items():
    try:
        frames.append(run_model(name, hf_id, prefix).set_index("語"))
    except Exception as e:
        print(f"※ {name} をスキップ: {type(e).__name__}: {e}")
        clear_cuda()
if len(frames) < 2:
    raise SystemExit("2モデル以上が必要です")
wide = pd.concat(frames, axis=1).reset_index()
used = [c.split("|")[0] for c in wide.columns if c.endswith("|z_差")]

def mean_of(key):
    return wide[[f"{n}|{key}" for n in used]].mean(axis=1)

zd = wide[[f"{n}|z_差" for n in used]].values
za = wide[[f"{n}|匿名_z" for n in used]].values
zb = wide[[f"{n}|プライバシー_z" for n in used]].values

# ---------------- 4. プロット用の集計表 ----------------
df = pd.DataFrame({"語": wide["語"], "分野": wide["語"].map(cat_of)})
df["匿名_cos_生"]        = mean_of("匿名_cos_生")
df["プライバシー_cos_生"]  = mean_of("プライバシー_cos_生")
df["匿名_cos_補正"]       = mean_of("匿名_cos_補正")
df["プライバシー_cos_補正"] = mean_of("プライバシー_cos_補正")
df["差_cos_補正"]        = df["匿名_cos_補正"] - df["プライバシー_cos_補正"]
df["匿名_z"]            = mean_of("匿名_z")
df["プライバシー_z"]      = mean_of("プライバシー_z")
df["z_差"]              = mean_of("z_差")
df["z差_最小"] = zd.min(1)
df["z差_最大"] = zd.max(1)
df["共通度z"] = np.minimum(za, zb).mean(1)
df["関連度z"] = np.maximum(za, zb).mean(1)

def judge(thr):
    anon = df["z差_最小"] > thr
    priv = df["z差_最大"] < -thr
    shared = (np.abs(zd).max(1) < 1.0) & (df["共通度z"] >= Z_SHARED_MIN)
    return np.select([anon, priv, shared], ["匿名寄り", "プライバシー寄り", "共通（接点）"], "不一致・中間")

df["判定"] = judge(Z_THR)
df["判定_緩"] = judge(Z_THR_SOFT)

# ---------------- 5. 画面表示 ----------------
print("\n" + "=" * 78)
print(f"【1】判定の件数（モデル: {', '.join(used)} / 基準: {REF_MODE}）")
print("=" * 78)
print(pd.DataFrame({f"厳しい(Z={Z_THR})": df["判定"].value_counts(),
                    f"緩い(Z={Z_THR_SOFT})": df["判定_緩"].value_counts()}).fillna(0).astype(int).to_string())

print("\n" + "=" * 78)
print("【2】分野別の平均（z得点。匿名_z・プライバシー_zは、各概念への近さ）")
print("=" * 78)
cat_tbl = (df.groupby("分野")
             .agg(語数=("語", "size"), 匿名_z=("匿名_z", "mean"), プライバシー_z=("プライバシー_z", "mean"),
                  z_差=("z_差", "mean"),
                  匿名寄り=("判定_緩", lambda s: (s == "匿名寄り").sum()),
                  プライバシー寄り=("判定_緩", lambda s: (s == "プライバシー寄り").sum()),
                  共通=("判定_緩", lambda s: (s == "共通（接点）").sum()))
             .sort_values("z_差", ascending=False).reset_index())
print(cat_tbl.to_string(index=False, float_format=lambda x: f"{x:+.2f}"))

print("\n" + "=" * 78)
print("【3】全体の指標")
print("=" * 78)
r_all = np.corrcoef(df["匿名_z"], df["プライバシー_z"])[0, 1]
r_noctl = np.corrcoef(df[df["分野"] != "対照（無関係語）"]["匿名_z"],
                      df[df["分野"] != "対照（無関係語）"]["プライバシー_z"])[0, 1]
print(f"匿名_z と プライバシー_z の相関: 全語={r_all:.3f} / 対照語を除く={r_noctl:.3f}")
print("  → 高いほど「2つの概念への近さが連動している（両方に近い語は両方に近い）」")
from itertools import combinations
for a, b in combinations(used, 2):
    rho = wide[f"{a}|z_差"].rank().corr(wide[f"{b}|z_差"].rank())
    print(f"モデル間の順位相関（z_差）{a} × {b}: {rho:.3f}")

def show(g, title, key, ascending, k=25):
    g = g.sort_values(key, ascending=ascending).head(k)
    print(f"\n▼ {title}  上位{len(g)}語")
    print(f"{'#':>3} {'語':<14}{'匿名z':>8}{'ﾌﾟﾗｲﾊﾞｼｰz':>11}{'z差':>8}  分野")
    for i, (_, r) in enumerate(g.iterrows(), 1):
        print(f"{i:>3} {r['語']:<14}{r['匿名_z']:>+8.2f}{r['プライバシー_z']:>+11.2f}{r['z_差']:>+8.2f}  {r['分野']}")

core = df[df["分野"] != "対照（無関係語）"]
show(core[core["判定_緩"] == "匿名寄り"], "匿名寄り（緩い基準）", "z差_最小", False)
show(core[core["判定_緩"] == "プライバシー寄り"], "プライバシー寄り（緩い基準）", "z差_最大", True)
show(core[core["判定_緩"] == "共通（接点）"], "共通（接点）", "共通度z", False)
ctl = df[df["分野"] == "対照（無関係語）"]
print(f"\n対照（無関係語）{len(ctl)}語の平均: 匿名_z={ctl['匿名_z'].mean():+.2f} / "
      f"プライバシー_z={ctl['プライバシー_z'].mean():+.2f}（0付近なら基準が機能している）")

# ---------------- 6. CSV出力 ----------------
out_dir = Path("/content/output"); out_dir.mkdir(exist_ok=True)
for f in out_dir.glob("*"):
    f.unlink()
enc = "utf-8-sig"
df.sort_values("z_差", ascending=False).to_csv(out_dir / "1_プロット用_モデル平均.csv", index=False, encoding=enc)
wide.merge(df[["語", "分野"]], on="語").to_csv(out_dir / "2_モデル別_詳細.csv", index=False, encoding=enc)
long_rows = []
for n in used:
    t = wide[["語"]].copy()
    t["分野"] = t["語"].map(cat_of)
    t["モデル"] = n
    for k in ["匿名_cos_生", "プライバシー_cos_生", "匿名_cos_補正", "プライバシー_cos_補正",
              "匿名_z", "プライバシー_z", "z_差"]:
        t[k] = wide[f"{n}|{k}"]
    long_rows.append(t)
pd.concat(long_rows).to_csv(out_dir / "3_プロット用_モデル別ロング形式.csv", index=False, encoding=enc)
cat_tbl.to_csv(out_dir / "4_分野別集計.csv", index=False, encoding=enc)
pd.DataFrame({"項目": ["語彙数", "参照語数", "基準", "使用モデル", "Z_THR", "Z_THR_SOFT", "Z_SHARED_MIN", "文型"],
              "値": [len(vocab), len(ref), REF_MODE, ", ".join(used), Z_THR, Z_THR_SOFT, Z_SHARED_MIN,
                     " | ".join(TEMPLATES)]}).to_csv(out_dir / "5_設定.csv", index=False, encoding=enc)

# ---------------- 7. 散布図（任意） ----------------
if MAKE_PLOT:
    try:
        import matplotlib
        import matplotlib.pyplot as plt
        from matplotlib import font_manager
        fp = None
        for cand in ["/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
                     "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc"]:
            if Path(cand).exists():
                fp = cand
        if fp is None:
            subprocess.run(["apt-get", "-y", "-qq", "install", "fonts-noto-cjk"],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            for cand in ["/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
                         "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc"]:
                if Path(cand).exists():
                    fp = cand
        if fp:
            font_manager.fontManager.addfont(fp)
            plt.rcParams["font.family"] = font_manager.FontProperties(fname=fp).get_name()
        fig, ax = plt.subplots(figsize=(11, 9))
        cats = list(df["分野"].unique())
        cmap = plt.get_cmap("tab20")
        for i, c in enumerate(cats):
            g = df[df["分野"] == c]
            ax.scatter(g["プライバシー_z"], g["匿名_z"], s=18, alpha=0.7, color=cmap(i % 20), label=f"{c}({len(g)})")
        lim = [min(df["プライバシー_z"].min(), df["匿名_z"].min()) - 0.5,
               max(df["プライバシー_z"].max(), df["匿名_z"].max()) + 0.5]
        ax.plot(lim, lim, "--", color="gray", lw=1)
        ax.set_xlim(lim); ax.set_ylim(lim)
        for _, r in df.reindex(df["z_差"].abs().sort_values(ascending=False).index).head(25).iterrows():
            ax.annotate(r["語"], (r["プライバシー_z"], r["匿名_z"]), fontsize=8, xytext=(3, 3), textcoords="offset points")
        ax.set_xlabel("プライバシーへの近さ（z得点）"); ax.set_ylabel("匿名への近さ（z得点）")
        ax.set_title("自前語彙の分布：対角線より上＝匿名寄り、下＝プライバシー寄り")
        ax.legend(fontsize=8, loc="lower right")
        fig.tight_layout()
        fig.savefig(out_dir / "6_散布図.png", dpi=150)
        plt.show()
    except Exception as e:
        print("散布図の作成をスキップ:", type(e).__name__, e)

print("\nCSV出力:")
for f in sorted(out_dir.glob("*")):
    print(" -", f)
try:
    from google.colab import files
    shutil.make_archive("/content/own_vocab_results", "zip", out_dir)
    files.download("/content/own_vocab_results.zip")
except Exception as e:
    print("自動ダウンロードをスキップ:", e)
print("\n分析完了")
