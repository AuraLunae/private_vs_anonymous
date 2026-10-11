import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import japanize_matplotlib
from adjustText import adjust_text

sns.set_theme(style="whitegrid", context="notebook")
japanize_matplotlib.japanize()
df = pd.read_csv("japanese.csv")
df.info()

sns.scatterplot(data=df, x="匿名_cos_補正", y="プライバシー_cos_生", hue="分野", palette="pastel")
plt.xlabel("匿名寄り")
plt.ylabel("プライバシー寄り")
plt.legend(loc='upper left', bbox_to_anchor=(1.05, 1))

top_t = df.nlargest(10, "匿名_cos_補正").reset_index()
top_p = df.nlargest(10, "プライバシー_cos_生").reset_index()
top_df = pd.concat([top_t, top_p], ignore_index=True)
print(top_df)

texts = []
# 抽出した10件にだけラベルをつける
for i in range(len(top_df)):
    t = plt.text(
        x=top_df["匿名_cos_補正"][i], 
        y=top_df["プライバシー_cos_生"][i], 
        s=top_df["語"][i], 
        fontsize=9
    )
    texts.append(t)

# 10件ならメモリ不足にならずに自動調整できる
adjust_text(texts, arrowprops=dict(arrowstyle='-', color='gray', lw=0.5))
plt.savefig('result.png', dpi=600, bbox_inches='tight')

plt.show()
