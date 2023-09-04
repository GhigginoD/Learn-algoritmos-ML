import matplotlib.pyplot as plt
import pandas as pd

base_de_dados = pd.read_csv("Pyplot/dataYoutubers.csv", delimiter=",", encoding="ISO-8859-1")

top_10_highest_subscribed = base_de_dados[["Youtuber", "subscribers", "category"]].head(
    10
)

plt.barh(
    top_10_highest_subscribed["Youtuber"],
    top_10_highest_subscribed["subscribers"],
    label="Number of Subscribers",
)
plt.xlabel("Subscribers in million")
plt.ylabel("Top_10_Youtubers")

plt.legend()
plt.show()
