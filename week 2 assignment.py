import pandas as pd

df = pd.read_csv("dataset 2.csv")
print(df.head())


print(df.shape)

print(df.columns)

#Numerical Features:

#UserID
#Age
#WatchHoursPerWeek
#DevicesUsed
#AdClicks
#MonthlySpend

#Categorical Features:

#Gender
#SubscriptionType
#FavoriteGenre
#SubscriptionRenewed

print(df.isnull().sum())
print(df["Age"].mean())

print(df["WatchHoursPerWeek"].mean())
print(df["MonthlySpend"].mean())
print(df["SubscriptionType"].value_counts())
renewed = (df["SubscriptionRenewed"]=="Yes").mean()*100
print(renewed)

from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

df["Gender"] = le.fit_transform(df["Gender"])
df["SubscriptionType"] = le.fit_transform(df["SubscriptionType"])
df["FavoriteGenre"] = le.fit_transform(df["FavoriteGenre"])
df["SubscriptionRenewed"] = le.fit_transform(df["SubscriptionRenewed"])


X = df.drop("SubscriptionRenewed", axis=1)
y = df["SubscriptionRenewed"]

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

from sklearn.tree import DecisionTreeClassifier

dt = DecisionTreeClassifier(random_state=42)
dt.fit(X_train, y_train)

from sklearn.metrics import accuracy_score

pred = dt.predict(X_test)
accuracy = accuracy_score(y_test, pred)

print(accuracy)

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, pred)
print(cm)
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
knn_pred = knn.predict(X_test)

from sklearn.metrics import accuracy_score

knn_acc = accuracy_score(y_test, knn_pred)

print(knn_acc)
from sklearn.linear_model import LinearRegression

X_reg = df.drop("MonthlySpend", axis=1)
y_reg = df["MonthlySpend"]

lr = LinearRegression()
lr.fit(X_reg, y_reg)
new_user = X_reg.iloc[[0]]
prediction = lr.predict(new_user)

print(prediction)
# Business Reflection Questions

# 1. Which factors appear to influence subscription renewal the most?
# Answer:
# The factors that influence subscription renewal the most are:
# - Monthly Spend
# - Watch Hours Per Week
# - Age
# - Ad Clicks
# - Subscription Type

# 2. Why is subscription renewal a classification problem?
# Answer:
# Subscription renewal is a classification problem because the output has
# two categories: Yes or No.

# 3. Why is monthly spending a regression problem?
# Answer:
# Monthly spending is a regression problem because it is a continuous
# numerical value that can take many different amounts.

# 4. Which algorithm performed better for renewal prediction?
# Answer:
# KNN performed better than Decision Tree because it achieved higher accuracy.

# 5. How could the platform use these predictions to improve customer retention?
# Answer:
# Netflix can:
# - Identify users likely to cancel subscriptions.
# - Offer personalized discounts and promotions.
# - Recommend suitable content.
# - Improve customer satisfaction.
# - Increase customer retention and revenue.

https://github.com/shreyakvdc-blip/WEEK-2-FINAL-

