# Gradient Boosting ও XGBoost সম্পূর্ণ টিউটোরিয়াল (A to Z)
### সহজ ভাষায়, গণিত ও কোড সহ

---

## অংশ ০: শুরুর আগে — বেসিক ধারণাগুলো ঝালাই

### ০.১ Decision Tree কী (এক লাইনে)

Decision Tree হলো একটা প্রশ্ন-উত্তর কাঠামো (যেমন "বয়স < 30?") — যা বার বার ডেটা ভাগ করতে করতে শেষে একটা leaf-এ prediction দেয়। একা একটা tree সাধারণত দুর্বল (weak learner) — হয় খুব সহজ (underfit) নয়তো খুব জটিল হয়ে ট্রেনিং ডেটা মুখস্থ করে ফেলে (overfit)।

### ০.২ "Weak Learner"-দের একত্র করার আইডিয়া

একটা মজার উপমা দিয়ে বোঝা যাক:

> ধরুন আপনি একটা রচনা লিখছেন। প্রথমবার লেখার পর একজন শিক্ষক ভুলগুলো ধরিয়ে দিলেন। আপনি সেই ভুলগুলো ঠিক করলেন। আবার আরেকজন শিক্ষক বাকি ভুলগুলো ধরিয়ে দিলেন। এভাবে একের পর এক "সংশোধন" যোগ করতে করতে রচনাটা ধীরে ধীরে নিখুঁত হয়ে ওঠে।

Boosting ঠিক এই কাজটাই করে — প্রতিটা নতুন tree আগের সব tree মিলিয়ে যে prediction দিয়েছে, তার **ভুলটুকু** ঠিক করার চেষ্টা করে। এটাই Gradient Boosting-এর মূল ভাবনা।

---

## অংশ ১: Gradient Boosting — ধাপে ধাপে গণিত (সহজ ভাষায়)

### ১.১ সহজ উদাহরণ দিয়ে শুরু (Regression)

ধরুন আমরা মানুষের বয়স predict করতে চাই। আমাদের কাছে ৩ জনের ডেটা:

| Person | Actual বয়স ($y$) |
|---|---|
| A | 20 |
| B | 30 |
| C | 40 |

**Step 1: সবচেয়ে সহজ prediction দিয়ে শুরু**

প্রথম মডেল $F_0$ শুধু গড় predict করে:
$$F_0(x) = \text{mean}(y) = \frac{20+30+40}{3} = 30$$

**Step 2: Residual (ভুল) হিসাব করা**

প্রতিটা person-এর জন্য ভুল = Actual − Predicted:

| Person | Actual | Predicted ($F_0$) | Residual |
|---|---|---|---|
| A | 20 | 30 | -10 |
| B | 30 | 30 | 0 |
| C | 40 | 30 | +10 |

**Step 3: একটা নতুন ছোট Tree ট্রেইন করা — কিন্তু Actual বয়সের উপর না, বরং এই Residual-এর উপর!**

এই নতুন tree ($h_1$) শিখবে input feature দিয়ে residual predict করতে। ধরুন এটা মোটামুটি এই residual-গুলোই predict করল (perfect fit ধরে নিলাম উদাহরণের সুবিধার জন্য): A → -10, B → 0, C → +10।

**Step 4: পুরনো prediction-এর সাথে নতুন tree-র prediction যোগ করা (একটা learning rate সহ)**

$$F_1(x) = F_0(x) + \nu \times h_1(x)$$

যদি $\nu$ (learning rate) = 0.1 হয়, তাহলে:
- A: $30 + 0.1 \times (-10) = 29$
- B: $30 + 0.1 \times 0 = 30$
- C: $30 + 0.1 \times 10 = 31$

দেখুন, prediction আগের চেয়ে actual value-র একটু কাছে গেছে! এই প্রসেস বারবার (M বার) চালানো হয় — প্রতিবার নতুন residual বের করে, নতুন tree ফিট করে, একটু একটু করে prediction ঠিক করা হয়।

**এটাই পুরো Gradient Boosting-এর মূল ভাব — খুব সহজ ভাষায়।**

### ১.২ কেন "Gradient" নাম? (গাণিতিক গভীরতা)

আমরা একটা loss function দিয়ে মাপি prediction কতটা ভুল। Regression-এ সাধারণত **Squared Error** ব্যবহার হয়:
$$L(y, F(x)) = \frac{1}{2}(y - F(x))^2$$

এখন মজার ব্যাপার হলো — এই loss-কে $F(x)$-এর সাপেক্ষে derivative (gradient) নিলে কী পাই দেখুন:
$$\frac{\partial L}{\partial F(x)} = -(y - F(x))$$

মানে **negative gradient = residual** ( $y - F(x)$ )। এই কারণেই যখন আমরা residual-এর উপর tree ফিট করি, তখন আসলে আমরা loss function-এর **negative gradient**-এর দিকে এগোচ্ছি — অর্থাৎ **Gradient Descent** করছি, কিন্তু parameter space-এ না, বরং **function space**-এ (একটা নতুন ফাংশন/tree যোগ করে করে loss কমানো হচ্ছে)। তাই নাম **Gradient Boosting**।
সাধারণভাবে যেকোনো loss function-এর জন্য (regression, classification, ranking — যেকোনো কিছু):

```math
r_{im}
=
-
\left.
\frac{\partial L(y_i, F(x_i))}
{\partial F(x_i)}
\right|_{F(x)=F_{m-1}(x)}
```
এই $r_{im}$-কে বলে **pseudo-residual**, আর নতুন tree $h_m(x)$ এটার উপর ফিট করা হয়।

### ১.৩ পূর্ণ অ্যালগরিদম (Formal)

**ইনপুট**: ডেটা $\{(x_i, y_i)\}_{i=1}^n$, loss function $L$, iteration সংখ্যা $M$

**Step 1** — Initialize:
$$F_0(x) = \arg\min_{\gamma} \sum_{i=1}^n L(y_i, \gamma)$$
(regression-এ এটা সাধারণত mean, classification-এ log-odds)

**Step 2** — প্রতিটা $m = 1$ থেকে $M$ পর্যন্ত:

(a) Pseudo-residual হিসাব করা:
$$r_{im} = -\left[\frac{\partial L(y_i, F(x_i))}{\partial F(x_i)}\right]_{F=F_{m-1}}, \quad i=1,\ldots,n$$

(b) একটা regression tree $h_m(x)$ ফিট করা $\{(x_i, r_{im})\}$ ডেটার উপর

(c) প্রতিটা leaf $j$-এর জন্য optimal output value বের করা:
$$\gamma_{jm} = \arg\min_{\gamma} \sum_{x_i \in R_{jm}} L(y_i, F_{m-1}(x_i) + \gamma)$$

(d) Model আপডেট:
$$F_m(x) = F_{m-1}(x) + \nu \sum_j \gamma_{jm} \mathbb{1}(x \in R_{jm})$$

**আউটপুট**: $F_M(x)$

### ১.৪ Classification-এ কীভাবে কাজ করে?

Binary classification-এ **Log Loss** ব্যবহার হয়:
$$L(y, F(x)) = \log(1 + e^{-2yF(x)}), \quad y \in \{-1, +1\}$$

এখানে $F(x)$ প্রেডিক্ট করে **log-odds**, আর probability বের করতে হয় Sigmoid দিয়ে:
$$p = \frac{1}{1+e^{-F(x)}}$$

এই loss-এর gradient হয়:
$$r_i = y_i - p_i$$

মানে classification-এও একই নীতি — actual label আর predicted probability-র পার্থক্যই "residual" যা পরের tree শেখে।

### ১.৫ Learning Rate ($\nu$) কেন জরুরি?

যদি আমরা $\nu = 1$ রাখি, তাহলে প্রতিটা tree পুরোপুরি residual ঠিক করার চেষ্টা করবে — দ্রুত কিন্তু ট্রেনিং ডেটায় overfit করার ঝুঁকি বেশি।

$\nu$ ছোট রাখলে (যেমন 0.01-0.1) প্রতিটা tree শুধু **অল্প একটু** অবদান রাখে, তাই অনেকগুলো tree লাগে ঠিকমতো শিখতে — কিন্তু generalization ভালো হয়। এটাকে বলে **Shrinkage**। এটা অনেকটা ছোট ছোট পায়ে হাঁটার মতো — বড় লাফ দিলে লক্ষ্য ছাড়িয়ে যাওয়ার (overshoot) সম্ভাবনা থাকে।

**Trade-off**: $\nu$ ছোট → বেশি tree লাগবে ($M$ বাড়াতে হবে) → বেশি সময় লাগবে, কিন্তু ফলাফল ভালো হয়।

---

## অংশ ২: XGBoost (Extreme Gradient Boosting)

XGBoost (Tianqi Chen, 2016) হলো Gradient Boosting-এর একটা **অপ্টিমাইজড, রেগুলারাইজড, ও ইঞ্জিনিয়ারিং-ভাবে অনেক উন্নত** ভার্সন। এটা মূল GBM-এর উপর কয়েকটা গুরুত্বপূর্ণ উন্নতি যোগ করে।

### ২.১ XGBoost-এর মূল উন্নতিগুলো (এক নজরে)

1. **Regularized Objective Function** — overfitting রোধে tree complexity-কে সরাসরি loss function-এ পেনাল্টি দেয়
2. **2nd Order Taylor Approximation** — শুধু gradient না, Hessian (2nd derivative)-ও ব্যবহার করে, ফলে optimization বেশি নির্ভুল ও দ্রুত converge করে
3. **Efficient Split Finding** — Approximate algorithm, weighted quantile sketch
4. **Sparsity-aware split finding** — missing value নিজে থেকেই handle করে
5. **System-level optimization** — parallel processing, cache-aware access, out-of-core computation (বড় ডেটার জন্য)

### ২.২ Regularized Objective Function (মূল গাণিতিক পার্থক্য)

সাধারণ Gradient Boosting শুধু loss কমানোর চেষ্টা করে। XGBoost objective-এ tree-র জটিলতার জন্য একটা **penalty term** যোগ করে:

$$\text{Obj} = \sum_{i=1}^n L(y_i, \hat{y}_i) + \sum_{k=1}^K \Omega(f_k)$$

যেখানে regularization term:
$$\Omega(f) = \gamma T + \frac{1}{2}\lambda \sum_{j=1}^T w_j^2$$

- $T$ = tree-র leaf সংখ্যা
- $w_j$ = $j$-তম leaf-এর output value (weight)
- $\gamma$ = প্রতিটা নতুন leaf বানানোর জন্য "খরচ" (কম leaf রাখতে উৎসাহ দেয়)
- $\lambda$ = leaf weight-এর উপর $L2$ regularization (weight-কে ছোট রাখতে উৎসাহ দেয়, ঠিক Ridge Regression-এর মতো)

**সহজ ভাষায়**: এটা মডেলকে বলে — "শুধু ডেটার সাথে ভালোভাবে মিলে গেলেই হবে না, তুমি সরল/simple-ও থাকতে হবে।" এটাই XGBoost-কে overfitting থেকে ভালোভাবে রক্ষা করে।

### ২.৩ 2nd Order Taylor Expansion — কেন ও কীভাবে

$m$-তম iteration-এ objective:
$$\text{Obj}^{(m)} = \sum_{i=1}^n L(y_i, F_{m-1}(x_i) + h_m(x_i)) + \Omega(h_m)$$

সরাসরি এই optimize করা কঠিন যেকোনো loss function-এর জন্য। তাই XGBoost Taylor series দিয়ে approximate করে (2nd order পর্যন্ত):

$$\text{Obj}^{(m)} \approx \sum_{i=1}^n \left[L(y_i, F_{m-1}(x_i)) + g_i h_m(x_i) + \frac{1}{2} f_i h_m(x_i)^2\right] + \Omega(h_m)$$

যেখানে:
$$g_i = \frac{\partial L(y_i, F_{m-1}(x_i))}{\partial F_{m-1}(x_i)} \quad \text{(Gradient — 1st derivative)}$$
$$f_i = \frac{\partial^2 L(y_i, F_{m-1}(x_i))}{\partial F_{m-1}(x_i)^2} \quad \text{(Hessian — 2nd derivative)}$$

**সহজ উপমা**: Gradient বলে দেয় "কোন দিকে এগোতে হবে", আর Hessian বলে দেয় "কতটা জোরে/দ্রুত এগোনো উচিত" (curvature)। এই দুটো তথ্য মিলিয়ে অনেক বেশি নির্ভুল এবং দ্রুত optimization হয় — এটাই Newton's Method-এর মূল ভাব, যা শুধু Gradient Descent-এর চেয়ে উন্নত।

### ২.৪ Optimal Leaf Weight ও Gain — Split কীভাবে বাছা হয়

Constant term বাদ দিয়ে, একটা নির্দিষ্ট tree structure $q$-এর জন্য objective simplify করলে:

$$\text{Obj} = \sum_{j=1}^T \left[G_j w_j + \frac{1}{2}(H_j + \lambda) w_j^2\right] + \gamma T$$

যেখানে $G_j = \sum_{i \in I_j} g_i$ ও $H_j = \sum_{i \in I_j} f_i$ (leaf $j$-এর সব sample-এর gradient/hessian-এর যোগফল)।

এই quadratic function-কে $w_j$-এর সাপেক্ষে minimize করলে (ডেরিভেটিভ = 0 বসিয়ে):

$$w_j^* = -\frac{G_j}{H_j + \lambda}$$

আর সেই optimal weight বসিয়ে minimum objective value:
$$\text{Obj}^* = -\frac{1}{2}\sum_{j=1}^T \frac{G_j^2}{H_j + \lambda} + \gamma T$$

এই মান-টাকেই ব্যবহার করা হয় **কোন split ভালো তা বাছাই করার জন্য** — একে বলে **Structure Score** (কম মান = ভালো tree structure)।

**Split করা উচিত কিনা যাচাই করার সূত্র (Gain):**

একটা node-কে Left ($L$) আর Right ($R$)-এ split করলে gain হয়:
$$\text{Gain} = \frac{1}{2}\left[\underbrace{\frac{G_L^2}{H_L+\lambda}}_{\text{left leaf score}} + \underbrace{\frac{G_R^2}{H_R+\lambda}}_{\text{right leaf score}} - \underbrace{\frac{(G_L+G_R)^2}{H_L+H_R+\lambda}}_{\text{split না করলে score}}\right] - \gamma$$

**সহজ ভাষায়**: split করার পর score (accuracy-জাতীয় কিছু) কতটা বাড়ল, তা থেকে split করার "খরচ" ($\gamma$) বাদ দেওয়া হয়। যদি Gain পজিটিভ হয়, তাহলে split করা লাভজনক; নাহলে সেখানেই tree growth থামিয়ে দেওয়া হয় (এটাকে বলে **pruning** — XGBoost আসলে "backward pruning" করে, মানে প্রথমে গভীরে split করে তারপর negative gain-ওয়ালা branch কেটে ফেলে)।

### ২.৫ Split খোঁজার পদ্ধতি: Exact vs Approximate

**Exact Greedy Algorithm**: প্রতিটা feature-এর প্রতিটা সম্ভাব্য value-এ Gain হিসাব করে সেরাটা বেছে নেয়। নির্ভুল কিন্তু ধীর, বিশেষ করে বড় ডেটাসেটে (পুরো ডেটা মেমরিতে লাগে)।

**Approximate Algorithm**: ডেটাকে percentile অনুযায়ী কয়েকটা bucket/bin-এ ভাগ করে (Weighted Quantile Sketch ব্যবহার করে, যেখানে weight = Hessian $f_i$, কারণ Hessian ভুলের "গুরুত্ব" বোঝায়), তারপর শুধু bin boundary-তে split চেক করে। বড় ডেটাসেটে অনেক দ্রুত।

### ২.৬ Sparsity-Aware Split Finding (Missing Value হ্যান্ডলিং)

Real-world ডেটায় প্রায়ই missing value থাকে। XGBoost প্রতিটা split-এর জন্য একটা **default direction** শেখে — training-এর সময় missing value-ওয়ালা sample-গুলোকে একবার left-এ, একবার right-এ পাঠিয়ে দেখে কোনটাতে বেশি Gain হয়, আর সেটাকেই "default" হিসেবে রেখে দেয়। ফলে ইউজারকে আলাদা করে imputation করতে হয় না।

### ২.৭ System Optimization (সংক্ষেপে)

- **Column Block / Cache-aware Access**: ডেটা column-wise sorted আকারে ব্লকে সংরক্ষণ করা হয়, যাতে split finding parallel হতে পারে এবং CPU cache efficiently ব্যবহার হয়।
- **Parallel Learning**: Tree structure sequential হলেও, একটা tree-র ভেতরে feature-wise split finding parallel-এ করা যায়।
- **Out-of-core computation**: ডেটা RAM-এ না ধরলে disk থেকে block-ভাবে পড়ে ট্রেনিং করতে পারে।

---

## অংশ ৩: Gradient Boosting বনাম XGBoost — মূল পার্থক্য সংক্ষেপে

| বিষয় | সাধারণ Gradient Boosting | XGBoost |
|---|---|---|
| Derivative ব্যবহার | শুধু 1st order (Gradient) | 1st ও 2nd order (Gradient + Hessian) |
| Regularization | নেই (বেসিক ভার্সনে) | আছে ($\gamma$, $\lambda$ — leaf সংখ্যা ও weight উভয়ে) |
| Missing value | নিজে থেকে হ্যান্ডল করে না | Sparsity-aware, নিজে থেকেই শেখে |
| স্পিড | তুলনামূলক ধীর | Parallel, cache-optimized, অনেক দ্রুত |
| Split finding | Exact only (সাধারণত) | Exact + Approximate (Weighted Quantile Sketch) |
| Pruning | Greedy (আগে থেকে না জেনে থামে) | Backward pruning (গভীরে গিয়ে negative gain কাটে) |
| Overfitting নিয়ন্ত্রণ | শুধু learning rate, depth, subsampling | উপরের সবকিছু + built-in regularization |

---

## অংশ ৪: কোড — Scratch থেকে বেসিক Gradient Boosting

গণিতটা আরও ভালোভাবে বুঝতে, নিচে একদম সাধারণ regression-এর জন্য Gradient Boosting-এর simplified implementation:

```python
import numpy as np
from sklearn.tree import DecisionTreeRegressor

class SimpleGradientBoosting:
    def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=3):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.trees = []
        self.initial_pred = None

    def fit(self, X, y):
        # Step 1: Initialize করা (mean দিয়ে)
        self.initial_pred = np.mean(y)
        F = np.full(y.shape, self.initial_pred)

        for m in range(self.n_estimators):
            # Step 2: Residual (negative gradient, squared-error loss-এর জন্য)
            residual = y - F

            # Step 3: নতুন tree residual-এর উপর ফিট করা
            tree = DecisionTreeRegressor(max_depth=self.max_depth)
            tree.fit(X, residual)
            self.trees.append(tree)

            # Step 4: Model আপডেট (learning rate সহ)
            F += self.learning_rate * tree.predict(X)

    def predict(self, X):
        F = np.full(X.shape[0], self.initial_pred)
        for tree in self.trees:
            F += self.learning_rate * tree.predict(X)
        return F

# ব্যবহার
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split

X, y = make_regression(n_samples=500, n_features=10, noise=10, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

gb = SimpleGradientBoosting(n_estimators=100, learning_rate=0.1, max_depth=3)
gb.fit(X_train, y_train)
preds = gb.predict(X_test)

from sklearn.metrics import mean_squared_error
print("MSE:", mean_squared_error(y_test, preds))
```

এই ছোট্ট কোডটাই আসলে Gradient Boosting-এর হৃদয়। sklearn-এর `GradientBoostingRegressor` বা XGBoost — সবই এই একই মূল ভাবনার উপর ভিত্তি করে তৈরি, শুধু optimization আর engineering আলাদা।

### ৪.১ Scikit-learn দিয়ে Gradient Boosting

```python
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier

gbr = GradientBoostingRegressor(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=3,
    subsample=0.8,       # প্রতিটা tree-তে random 80% ডেটা ব্যবহার (Stochastic GB)
    random_state=42
)
gbr.fit(X_train, y_train)
preds = gbr.predict(X_test)
```

### ৪.২ XGBoost দিয়ে কোড (পূর্ণ)

```bash
pip install xgboost
```

```python
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer

data = load_breast_cancer()
X, y = data.data, data.target
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Scikit-learn API — XGBClassifier
clf = XGBClassifier(
    n_estimators=300,
    max_depth=4,
    learning_rate=0.05,     # eta, $\nu$
    reg_lambda=1.0,         # L2 regularization ($\lambda$)
    gamma=0.1,              # minimum split loss (leaf তৈরির খরচ, উপরের $\gamma$)
    subsample=0.8,
    colsample_bytree=0.9,
    eval_metric='logloss',
    early_stopping_rounds=20,
    random_state=42
)

clf.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    verbose=50
)

preds = clf.predict(X_val)
proba = clf.predict_proba(X_val)[:, 1]
```

---

## অংশ ৫: গুরুত্বপূর্ণ হাইপারপ্যারামিটার (গণিতসহ ব্যাখ্যা)

| প্যারামিটার | সূত্রে কোথায় | প্রভাব |
|---|---|---|
| `eta` / `learning_rate` ($\nu$) | $F_m = F_{m-1} + \nu \cdot h_m$ | কম মান → বেশি tree লাগবে, কিন্তু ভালো generalization |
| `max_depth` | Tree-র গঠন | বেশি গভীরতা → বেশি জটিল, overfit ঝুঁকি |
| `n_estimators`/`num_boost_round` | $M$ | বেশি tree → বেশি ক্ষমতা, কিন্তু বেশি সময় ও overfit ঝুঁকি (early stopping দিয়ে নিয়ন্ত্রণ করুন) |
| `gamma` | Gain সূত্রে $\gamma$ | বেশি → conservative tree (কম split হবে) |
| `lambda`/`reg_lambda` | $w_j^* = -G_j/(H_j+\lambda)$ | বেশি → leaf weight ছোট হয়, underfit-এর দিকে টানে |
| `alpha`/`reg_alpha` | $L1$ regularization (sparsity আনে) | কিছু leaf weight একদম 0 করে দিতে পারে |
| `subsample` | Row sampling | কম → variance কমে, randomness বাড়ে (bagging-এর মতো প্রভাব) |
| `colsample_bytree` | Feature sampling | কম → feature-এর উপর নির্ভরতা কমে, Random Forest-এর ছোঁয়া |
| `min_child_weight` | একটা leaf-এ ন্যূনতম $\sum H_i$ | বেশি → conservative, ছোট leaf তৈরি হতে বাধা দেয় |

---

## অংশ ৬: Overfitting চেনা ও ঠেকানো

**Overfitting চেনার লক্ষণ**: Training loss ক্রমাগত কমছে, কিন্তু validation loss একটা পয়েন্টের পর বাড়তে শুরু করেছে।

```python
# Learning curve visualize করে দেখুন (clf = XGBClassifier অবজেক্ট)
results = clf.evals_result()
import matplotlib.pyplot as plt
plt.plot(results['validation_0']['logloss'], label='validation')
plt.plot(results['validation_1']['logloss'], label='train')  # যদি দুটো eval set দেন
plt.legend()
plt.show()
```

**সমাধানের ক্রম (সহজ থেকে জটিল):**
1. `early_stopping_rounds` ব্যবহার করুন — validation loss বাড়া শুরু করলেই থামিয়ে দিন
2. `learning_rate` কমান, `n_estimators` বাড়ান
3. `max_depth` কমান
4. `subsample`, `colsample_bytree` কমান (0.7-0.9 রেঞ্জে)
5. `gamma`, `lambda`, `alpha` বাড়িয়ে regularization বাড়ান
6. `min_child_weight` বাড়ান

---

## অংশ ৭: সংক্ষিপ্ত সারাংশ (Cheat Sheet)

- **Gradient Boosting-এর মূল ভাব**: বারবার residual (ভুল) হিসাব করে, সেই ভুলের উপর নতুন ছোট tree ফিট করে, ধীরে ধীরে (learning rate দিয়ে নিয়ন্ত্রিতভাবে) prediction ঠিক করা।
- **গাণিতিক ভিত্তি**: এটা function space-এ gradient descent — negative gradient = residual (squared error loss-এর ক্ষেত্রে)।
- **XGBoost** এই একই ভিত্তির উপর দাঁড়িয়ে কিন্তু:
  - 2nd order Taylor expansion (Gradient + Hessian) ব্যবহার করে বেশি নির্ভুল optimization করে,
  - Objective function-এ সরাসরি regularization ($\gamma$, $\lambda$) যোগ করে overfitting কমায়,
  - Missing value নিজে নিজে হ্যান্ডল করে,
  - System-level optimization (parallelism, cache-efficiency) দিয়ে অনেক দ্রুত কাজ করে।
- **মূল সূত্র মনে রাখার মতো**:
  - Leaf weight: $w_j^* = -\dfrac{G_j}{H_j+\lambda}$
  - Split Gain: $\text{Gain} = \frac{1}{2}\left[\dfrac{G_L^2}{H_L+\lambda} + \dfrac{G_R^2}{H_R+\lambda} - \dfrac{(G_L+G_R)^2}{H_L+H_R+\lambda}\right] - \gamma$

---

চাইলে আমি পরবর্তী ধাপে **LightGBM/CatBoost বনাম XGBoost-এর গাণিতিক Gain formula পাশাপাশি তুলনা** করে দেখাতে পারি (যেহেতু আগে ওই দুটোর টিউটোরিয়ালও করেছি), অথবা একটা real dataset দিয়ে XGBoost-এর সম্পূর্ণ end-to-end প্রজেক্ট (EDA → feature engineering → tuning → deployment) বানিয়ে দিতে পারি।
