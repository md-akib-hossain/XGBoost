# LightGBM ও CatBoost সম্পূর্ণ টিউটোরিয়াল (A to Z)
### সহজ ভাষায়, গণিত ও কোড সহ

---

## অংশ ০: শুরুর আগে — এরা কোথা থেকে এল?

আগের টিউটোরিয়ালে আমরা দেখেছি Gradient Boosting-এর মূল ভাব: বারবার residual (ভুল) হিসাব করে, সেই ভুলের উপর নতুন tree ফিট করে prediction ঠিক করা। আর XGBoost দেখেছি — যেটা এই একই ভাবের উপর regularization আর 2nd-order গণিত (Gradient + Hessian) যোগ করে আরও শক্তিশালী করেছে।

**LightGBM** আর **CatBoost** — এই দুটোও XGBoost-এর মতোই GBDT (Gradient Boosted Decision Tree) পরিবারের সদস্য। এরা একই মূল গণিত (Gradient + Hessian, Gain formula) ব্যবহার করে, কিন্তু প্রত্যেকে একটা নির্দিষ্ট **সমস্যা** সমাধানের জন্য ডিজাইন করা হয়েছে:

> **LightGBM** (Microsoft) → "ডেটাসেট বিশাল, ট্রেনিং যেন **সবচেয়ে দ্রুত** হয়"
> **CatBoost** (Yandex) → "ডেটায় প্রচুর categorical column আছে, আর **overfitting যেন না হয়**"

দুটোই XGBoost-এর "leaf weight ও Gain" গণিত থেকে শুরু করে, কিন্তু tree কীভাবে বানানো হয় সেই ইঞ্জিনিয়ারিংটা সম্পূর্ণ আলাদা। চলুন প্রতিটা একদম সহজ উদাহরণ দিয়ে বুঝি।

---

## অংশ ১: LightGBM — সহজ ভাষায়

### ১.১ সমস্যাটা কী ছিল?

ধরুন আপনার কাছে ১ কোটি row আর ১০০০ column-এর একটা ডেটাসেট আছে। সাধারণ Gradient Boosting/XGBoost-এর মতো পদ্ধতিতে প্রতিটা tree বানাতে প্রতিটা feature-এর **প্রতিটা সম্ভাব্য split point** চেক করতে হয় — যেটা এত বড় ডেটায় ভীষণ ধীর।

LightGBM-এর ইঞ্জিনিয়াররা ভাবলেন — "আমরা কি কম সময়ে প্রায় একই ফলাফল পেতে পারি, যদি:
1. Tree-টাকে একটু স্মার্টভাবে বানাই (leaf-wise)?
2. সব ডেটাপয়েন্ট চেক না করে গুরুত্বপূর্ণগুলো বেশি খেয়াল করি (GOSS)?
3. প্রতিটা value না দেখে, কিছু bucket/bin-এ ভাগ করে দেখি (Histogram)?"

এই তিনটা আইডিয়াই LightGBM-কে দ্রুত করে তোলে। একে একে দেখি।

### ১.২ Leaf-wise Growth — সহজ উদাহরণ দিয়ে

ধরুন আপনার একটা গাছের ৪টা পাতা (leaf) বানানোর বাজেট আছে। দুইভাবে বানানো যায়:

**Level-wise (পুরনো পদ্ধতি, XGBoost-এর ডিফল্ট)**: প্রথমে root split করে ২টা leaf বানাবে, তারপর **দুটো leaf-কেই** একসাথে আরেকবার split করে ৪টা বানাবে — এমনকি যদি একটা leaf split করার তেমন দরকারই না থাকে।

```
        Root
       /    \
    Leaf   Leaf      <- Level 1 (দুটোই split হবে, ভালো/খারাপ না দেখে)
    /  \   /  \
   L1  L2 L3  L4      <- Level 2
```

**Leaf-wise (LightGBM-এর পদ্ধতি)**: প্রতিবার শুধু **সেই leaf-টাকে split করবে যেটা split করলে সবচেয়ে বেশি লাভ (Gain) হবে** — লেভেল মিলিয়ে না, বরং সবচেয়ে "প্রয়োজনীয়" জায়গাতেই মনোযোগ দেয়।

```
        Root
       /    \
    Leaf    Leaf*  <- এই leaf-টায় সবচেয়ে বেশি Gain, তাই এটাই split হবে
            /   \
          Leaf  Leaf*  <- আবার সবচেয়ে বেশি Gain কোথায় আছে খুঁজবে
```

**গণিত**: XGBoost অংশে আমরা শিখেছিলাম Gain-এর সূত্র:
$$\text{Gain} = \frac{1}{2}\left[\frac{G_L^2}{H_L+\lambda} + \frac{G_R^2}{H_R+\lambda} - \frac{(G_L+G_R)^2}{H_L+H_R+\lambda}\right] - \gamma$$

LightGBM প্রতিটা candidate leaf $j$-এর জন্য এই Gain হিসাব করে, আর সর্বোচ্চ Gain-ওয়ালা leaf-টাকে বেছে নেয়:
$$j^* = \arg\max_j \text{Gain}(j)$$

**ফলাফল**: একই সংখ্যক leaf দিয়ে leaf-wise গাছ সাধারণত level-wise গাছের চেয়ে বেশি accurate হয় (কারণ effort গুরুত্বপূর্ণ জায়গায় খরচ হয়)। **কিন্তু** — ছোট ডেটাসেটে এই পদ্ধতি একটা দিকে বেশি গভীর ও সরু (deep, narrow) গাছ বানিয়ে ফেলতে পারে, যা overfit করে। তাই `num_leaves` আর `min_data_in_leaf` দিয়ে এটা নিয়ন্ত্রণ করতে হয়।

### ১.৩ Histogram-based Splitting — সহজ উদাহরণ

ধরুন একটা feature-এর মান আছে: 2.3, 2.35, 2.4, 2.45, ..., 99.8 (হাজার হাজার unique value)। Exact পদ্ধতিতে প্রতিটা value-তে split চেক করতে হবে — অনেক সময় লাগবে।

LightGBM এর বদলে করে এইভাবে:

**Step 1**: পুরো রেঞ্জটাকে ছোট ছোট **bucket/bin**-এ ভাগ করে (ডিফল্ট 255টা bin)। যেমন: [0-5], [5-10], [10-15], ... ইত্যাদি।

**Step 2**: প্রতিটা bin-এর জন্য একটা হিসাব রাখে — সেই bin-এ কতগুলো sample পড়ল, আর তাদের gradient ($g$) ও hessian ($f$) যোগফল কত।

**Step 3**: Split খোঁজার সময় প্রতিটা raw value না দেখে, শুধু **bin-এর সীমানায়** (boundary) Gain চেক করে।

**তুলনা**: এতে জটিলতা কমে $O(n)$ থেকে $O(\text{bins})$-এ (যেখানে bins সংখ্যা সাধারণত মাত্র 255, আর $n$ হতে পারে লক্ষ লক্ষ)।

একটা বাড়তি চালাকি আছে — **Histogram Subtraction**:
$$\text{Sibling histogram} = \text{Parent histogram} - \text{অন্য Sibling-এর histogram}$$

মানে একটা sibling node-এর histogram বানাতে নতুন করে গণনা লাগে না — parent-এর histogram থেকে ভাইয়ের histogram বিয়োগ দিলেই হয়ে যায়! এতে computation প্রায় অর্ধেক কমে যায়।

### ১.৪ GOSS (Gradient-based One-Side Sampling) — সহজ উদাহরণ

মূল প্রশ্ন: "ট্রেনিংয়ের সময় কি সব ডেটাপয়েন্ট সমান গুরুত্বপূর্ণ?"

উত্তর: না। ধরুন ক্লাসে একজন শিক্ষক homework check করছেন:
- যেসব ছাত্রছাত্রীর উত্তর এখনো অনেক ভুল (বড় gradient/error) — তাদের **বিশেষ মনোযোগ** দরকার।
- যেসব ছাত্রছাত্রী প্রায় ঠিকভাবেই করছে (ছোট gradient) — তাদের প্রতিবার নতুন করে চেক করার দরকার নেই, মাঝেমধ্যে একটু চেক করলেই চলবে।

GOSS ঠিক এই আইডিয়া প্রয়োগ করে:

**Step 1**: সব sample-কে তাদের gradient-এর absolute value অনুযায়ী sort করে।

**Step 2**: সবচেয়ে বড় gradient-ওয়ালা উপরের $a\%$ (যেমন 20%) সব **রেখে দেয়**।

**Step 3**: বাকি ছোট-gradient sample-গুলো থেকে randomly মাত্র $b\%$ (যেমন 10%) sample **বেছে নেয়**।

**Step 4**: কিন্তু একটা সমস্যা — যদি ছোট-gradient sample কমিয়ে ফেলি, তাহলে ডেটার আসল distribution পাল্টে যাবে (bias তৈরি হবে)। এটা ঠিক করতে ছোট-gradient sample-গুলোর ওজন বাড়িয়ে দেওয়া হয়:
$$\text{weight} = \frac{1-a}{b}$$

এভাবে ট্রেনিং ডেটার একটা বড় অংশ বাদ দিয়েও (স্পিড বাড়ে) মূল distribution প্রায় ঠিক থাকে (accuracy তেমন কমে না)।

### ১.৫ EFB (Exclusive Feature Bundling) — সহজ উদাহরণ

ধরুন আপনার ডেটাসেটে one-hot encoded categorical column আছে: `is_dhaka`, `is_chittagong`, `is_khulna` ইত্যাদি। একটা row-তে এই কলামগুলোর মধ্যে **শুধু একটাই** কখনো 1 হবে, বাকিগুলো সবসময় 0 (কারণ একজন মানুষ একসাথে দুই শহরের হতে পারে না)।

এই ধরনের feature-কে বলে **mutually exclusive**। LightGBM এই ধরনের feature-গুলোকে চিনে নিয়ে **একটা একক feature**-এ বান্ডিল করে ফেলে (যেমন একটা কলামেই encode করে দেয় — dhaka=1, chittagong=2, khulna=3)।

**ফলাফল**: হাজার হাজার sparse column একসাথে হয়ে অনেক কম সংখ্যক feature হয়ে যায় — histogram বানানোর সময়/মেমরি অনেক কমে যায়, কিন্তু তথ্য হারায় না।

### ১.৬ LightGBM কোড — একদম শুরু থেকে

```bash
pip install lightgbm
```

```python
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer
import lightgbm as lgb

data = load_breast_cancer()
X, y = data.data, data.target
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

clf = LGBMClassifier(
    n_estimators=200,
    num_leaves=31,           # leaf-wise গাছে সর্বোচ্চ leaf সংখ্যা
    learning_rate=0.05,
    max_depth=-1,            # -1 মানে কোনো সীমা নেই (leaf-wise-এ num_leaves দিয়েই মূলত নিয়ন্ত্রণ হয়)
    subsample=0.8,           # প্রতি iteration-এ 80% row ব্যবহার
    colsample_bytree=0.9,    # প্রতি tree-তে 90% feature ব্যবহার (randomness)
    random_state=42
)

clf.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    eval_metric='binary_logloss',
    callbacks=[lgb.early_stopping(stopping_rounds=20)]
)

y_pred = clf.predict(X_val)
y_pred_proba = clf.predict_proba(X_val)[:, 1]
```

### ১.৭ LightGBM হাইপারপ্যারামিটার — সহজ ব্যাখ্যা

| প্যারামিটার | সহজ ভাষায় কী করে | Overfit-এর সাথে সম্পর্ক |
|---|---|---|
| `num_leaves` | গাছে সর্বোচ্চ কয়টা "সিদ্ধান্ত বিন্দু" থাকবে | বেশি → জটিল গাছ → overfit ঝুঁকি |
| `max_depth` | গাছ কতটা গভীর হতে পারবে | কম → সরল গাছ → underfit ঝুঁকি |
| `min_data_in_leaf` | একটা leaf-এ ন্যূনতম কতগুলো sample থাকতে হবে | কম → ছোট ছোট leaf তৈরি হয় → overfit |
| `learning_rate` | প্রতিটা গাছ কতটা "জোরে" শেখাবে | বেশি → দ্রুত কিন্তু ভুল ঠিক করার সুযোগ কম |
| `feature_fraction` | প্রতি গাছে কত শতাংশ feature দেখবে | কম → randomness বেশি → overfit কমে |

**সহজ নিয়ম**: `num_leaves` বাড়ালে `min_data_in_leaf`-ও বাড়ান, নইলে গাছ ছোট ছোট leaf বানিয়ে ট্রেনিং ডেটা প্রায় মুখস্থ করে ফেলবে।

---

## অংশ ২: CatBoost — সহজ ভাষায়

### ২.১ সমস্যাটা কী ছিল?

ধরুন আপনার ডেটাসেটে একটা column আছে `city` — যার মান হতে পারে ঢাকা, চট্টগ্রাম, খুলনা, রাজশাহী... এরকম হাজারো শহরের নাম। এই column-টাকে সংখ্যায় রূপান্তর (encode) করতে হবে, কারণ মডেল টেক্সট বোঝে না।

**সাধারণ সমাধান — One-Hot Encoding**: প্রতিটা শহরের জন্য একটা আলাদা কলাম (`is_dhaka`, `is_chittagong`...)। কিন্তু হাজার শহর মানে হাজার নতুন কলাম — মেমরি আর স্পিড দুটোই নষ্ট হয়।

**আরেকটা সমাধান — Target/Mean Encoding**: প্রতিটা শহরকে সেই শহরের গড় target value দিয়ে replace করা। যেমন ঢাকার গড় "loan default rate" যদি 0.3 হয়, তাহলে `city=ঢাকা` কে 0.3 দিয়ে replace করা। এটা সহজ এবং কার্যকর মনে হয়, কিন্তু এখানে একটা **লুকানো সমস্যা** আছে।

### ২.২ Target Leakage সমস্যা — সহজ উদাহরণ দিয়ে বোঝা

ধরুন `city=ঢাকা`-র মাত্র ৩টা row আছে ডেটাসেটে:

| Row | City | Target (loan default?) |
|---|---|---|
| 1 | ঢাকা | 1 |
| 2 | ঢাকা | 0 |
| 3 | ঢাকা | 1 |

Row 1-কে encode করার সময় সাধারণ Target Encoding করবে: গড় = $(1+0+1)/3 = 0.67$।

**সমস্যাটা লক্ষ্য করুন**: Row 1-এর নিজের target value (যেটা 1) **নিজেই** এই গড় হিসাবের মধ্যে ঢুকে গেছে! মানে মডেল Row 1-কে encode করার সময় "আসলে Row 1-এর উত্তরটাও একটু দেখে নিয়েছে" — এটাকে বলে **Target Leakage**।

**ফলাফল**: ট্রেনিং সময়ে মডেল দেখতে অনেক ভালো পারফর্ম করে (কারণ সে আসলে উত্তর একটু "টুকে" ফেলেছে), কিন্তু নতুন/test ডেটায় (যেখানে এই সুবিধা নেই) পারফরম্যান্স হঠাৎ খারাপ হয়ে যায়। এটাকে বলে **Prediction Shift**।

### ২.৩ সমাধান: Ordered Target Statistics — সহজ উদাহরণ

CatBoost-এর আইডিয়াটা এরকম:

> "একটা row-কে encode করার সময়, শুধু তার **আগে** যেসব row দেখেছি (একটা কাল্পনিক সময়-ক্রম অনুযায়ী), তাদের তথ্যই ব্যবহার করব — নিজের বা ভবিষ্যতের কোনো তথ্য না।"

এটা অনেকটা এরকম — পরীক্ষার হলে একজন ছাত্র শুধু তার **আগে যারা পরীক্ষা দিয়ে গেছে** তাদের অভিজ্ঞতা থেকে শিখতে পারবে, নিজের উত্তরপত্র দেখে না।

**ধাপে ধাপে দেখি একই উদাহরণে:**

CatBoost প্রথমে ডেটাকে randomly একটা ক্রমে সাজায় (permutation), ধরি ক্রম হলো: Row 2 → Row 1 → Row 3।

- **Row 2 (ক্রমে প্রথম)**: এর আগে কোনো ঢাকার row নেই, তাই শুধু prior (গড় global target, ধরি 0.5) ব্যবহার হয়।
- **Row 1 (ক্রমে দ্বিতীয়)**: এর আগে শুধু Row 2 দেখা গেছে (target=0), তাই এর encoding হবে Row 2-এর target-এর উপর ভিত্তি করে (0-এর কাছাকাছি একটা মান, prior-এর সাথে মিশিয়ে)।
- **Row 3 (ক্রমে তৃতীয়)**: এর আগে Row 2 (0) ও Row 1 (1) দেখা গেছে, তাই এর encoding = এই দুটোর গড়ের কাছাকাছি (prior সহ smoothed)।

**লক্ষ্য করুন**: কোনো row-ই কখনো নিজের target value ব্যবহার করেনি নিজেকে encode করতে! Leakage বন্ধ।

**গাণিতিক সূত্র:**
$$\hat{x}_i^k = \frac{\sum_{j=1}^{p-1} [x_{\sigma(j)}^k = x_{\sigma(p)}^k] \cdot y_{\sigma(j)} + a \cdot P}{\sum_{j=1}^{p-1} [x_{\sigma(j)}^k = x_{\sigma(p)}^k] + a}$$

এই সূত্রটা দেখতে জটিল মনে হলেও আসলে খুব সহজ কথা বলে:

- $p$ = বর্তমান row-এর অবস্থান (position) permutation-এ
- $[x_{\sigma(j)}^k = x_{\sigma(p)}^k]$ মানে: "আগের row-টার category কি আমার category-র সাথে মিলে?" (মিললে 1, না মিললে 0)
- $y_{\sigma(j)}$ = সেই আগের row-এর target value
- $a$ = কতটা "prior"-এর উপর ভরসা করব (smoothing) — যদি একই category-র খুব কম row আগে দেখা যায়, তাহলে $a$ বেশি প্রভাব ফেলবে
- $P$ = prior (সাধারণত পুরো ডেটাসেটের গড় target)

**সহজ ভাষায় পুরো সূত্র**: "আমার category-র মতো আগের যত row দেখেছি তাদের target-এর গড় নাও, কিন্তু যদি এরকম row কম থাকে তাহলে গ্লোবাল গড়ের (prior) দিকে একটু ঝুঁকে থাকো (Bayesian smoothing)।"

CatBoost এটা **একাধিক random permutation** দিয়ে করে (একটা permutation-এ ভাগ্যক্রমে ভুল estimate হয়ে গেলে অন্যগুলো সেটা ব্যালেন্স করে), যাতে ফলাফল স্থিতিশীল হয়।

### ২.৪ Ordered Boosting — একই নীতি Tree Building-এও

শুধু categorical encoding না, একই "leakage" সমস্যা আসলে **gradient হিসাবেও** লুকিয়ে থাকে। সাধারণ Gradient Boosting-এ:

$$r_i = -\left[\frac{\partial L(y_i, F_{m-1}(x_i))}{\partial F_{m-1}(x_i)}\right]$$

এখানে $F_{m-1}$ মডেলটা কিন্তু **$x_i$ দিয়েই ট্রেইন হয়েছিল** (একই ডেটাসেটের অংশ)। তাই $x_i$-এর gradient হিসাব করার সময় model তার নিজের সম্পর্কে "আগে থেকে কিছুটা জেনে" ফেলে — এটাও এক ধরনের (কম severe হলেও) leakage।

**CatBoost-এর সমাধান**: একই permutation-নীতি এখানেও প্রয়োগ করা। $i$-তম sample-এর gradient হিসাব করা হয় এমন একটা মডেল $M_{i-1}$ দিয়ে যেটা **শুধু permutation-এ $i$-এর আগের sample-গুলো** দিয়ে ট্রেইন হয়েছে — কখনো $i$ নিজে সেই মডেল ট্রেনিং-এ অংশ নেয়নি।

$$g_i = -\left[\frac{\partial L(y_i, M_{i-1}(x_i))}{\partial M_{i-1}(x_i)}\right]$$

এতে gradient estimate পুরোপুরি "unbiased" হয় — অনেকটা প্রতিটা sample-কে সত্যিকারের "নতুন/unseen" ডেটার মতো ট্রিট করার মতো, এমনকি ট্রেনিং-এর সময়ও। এটা কম্পিউটেশনালি ভারী মনে হলেও CatBoost efficient কৌশলে (একসাথে একাধিক permutation ট্র্যাক করে, incremental আপডেট করে) এটা সম্ভব করে।

### ২.৫ Symmetric (Oblivious) Trees — সহজ উদাহরণ

CatBoost-এর গাছ একটু অন্যরকম — **Symmetric Tree**। এর নিয়ম: গাছের একই depth (স্তর)-এর সব node-এ **হুবহু একই প্রশ্ন (feature + threshold)** জিজ্ঞেস করা হয়।

```
              [বয়স < 30?]                <- Level 1: সব জায়গায় একই প্রশ্ন
           /              \
     [আয় < 50k?]      [আয় < 50k?]        <- Level 2: আবার সব জায়গায় একই প্রশ্ন
      /      \           /      \
   Leaf1   Leaf2      Leaf3   Leaf4
```

সাধারণ tree-তে বাম আর ডান শাখায় ভিন্ন ভিন্ন প্রশ্ন থাকতে পারত (যেমন বাম দিকে "আয় < 50k?" আর ডান দিকে "শিক্ষা কী?")। কিন্তু Symmetric Tree-তে পুরো লেভেল জুড়ে একটাই প্রশ্ন।

**কেন এটা ভালো?**

1. **Regularization হিসেবে কাজ করে**: গাছ কম "স্বাধীন" (flexible), তাই ডেটা কম মুখস্থ করতে পারে — overfitting কম হয়।
2. **Prediction অবিশ্বাস্য দ্রুত**: যেহেতু প্রতিটা লেভেলে প্রশ্ন একই, তাই একটা নতুন sample-এর জন্য leaf খুঁজে বের করা মানে শুধু প্রতিটা লেভেলের উত্তর (হ্যাঁ/না = 1/0) একটা বাইনারি নম্বরে পরিণত করা:

$$\text{leaf\_index} = \sum_{l=1}^{d} b_l \cdot 2^{l-1}, \quad b_l \in \{0,1\}$$

এটা একদম bit-operation-এর মতো সহজ ও দ্রুত হিসাব — তাই CatBoost-এর inference (prediction) স্পিড খুব ভালো।

### ২.৬ CatBoost কোড — একদম শুরু থেকে

```bash
pip install catboost
```

```python
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
import pandas as pd

df = pd.read_csv('data.csv')
cat_features = ['city', 'category', 'gender']  # যেসব কলাম categorical

X = df.drop('target', axis=1)
y = df['target']
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

clf = CatBoostClassifier(
    iterations=500,
    depth=6,
    learning_rate=0.05,
    loss_function='Logloss',
    eval_metric='AUC',
    early_stopping_rounds=30,
    verbose=50,
    random_state=42
)

# cat_features সরাসরি fit()-এ দিলেই হয় — আলাদা Pool বানাতে হয় না
clf.fit(X_train, y_train, cat_features=cat_features, eval_set=(X_val, y_val))

preds = clf.predict(X_val)
proba = clf.predict_proba(X_val)[:, 1]
```

**লক্ষ্য করুন** — `city`, `category`, `gender` কলামগুলোকে আমরা কোনো manual encoding করিনি! শুধু `cat_features` লিস্টে নাম দিয়ে দিয়েছি, আর CatBoost নিজেই Ordered Target Statistics প্রয়োগ করে এদের হ্যান্ডল করেছে। এটাই CatBoost ব্যবহারের সবচেয়ে বড় সুবিধা — কম কোড, কম ভুলের সুযোগ।

### ২.৭ CatBoost হাইপারপ্যারামিটার — সহজ ব্যাখ্যা

| প্যারামিটার | সহজ ভাষায় কী করে |
|---|---|
| `iterations` | মোট কয়টা গাছ বানাবে |
| `depth` | প্রতিটা Symmetric Tree কত গভীর হবে (সাধারণত 4-10) |
| `learning_rate` | প্রতিটা গাছ কতটা জোরে শেখাবে |
| `l2_leaf_reg` | leaf-এর output value-কে ছোট রাখার জন্য পেনাল্টি (overfit কমায়) |
| `cat_features` | কোন কলামগুলো categorical, সেই লিস্ট |
| `one_hot_max_size` | কম unique value-ওয়ালা categorical column সরাসরি one-hot হয়ে যাবে, বেশি হলে Ordered TS ব্যবহার হবে |
| `boosting_type` | `Ordered` (ছোট ডেটায় ভালো, leakage কম) বনাম `Plain` (বড় ডেটায় দ্রুত) |

---

## অংশ ৩: তিনটা মডেলের পাশাপাশি তুলনা (LightGBM vs CatBoost vs XGBoost)

| বৈশিষ্ট্য | LightGBM | CatBoost | XGBoost |
|---|---|---|---|
| মূল সমস্যা সমাধান করে | স্পিড (বড় ডেটা) | Categorical feature + Overfitting | সাধারণ, সুষম পারফরম্যান্স |
| Tree বানানোর ধরন | Leaf-wise (সবচেয়ে লাভজনক leaf আগে split) | Symmetric (একই level-এ একই প্রশ্ন) | Level-wise (ডিফল্ট) |
| Categorical হ্যান্ডলিং | মৌলিক সাপোর্ট আছে | সেরা (Ordered TS দিয়ে) | নেই (manual encode লাগে) |
| ট্রেনিং স্পিড | সবচেয়ে দ্রুত | তুলনামূলক ধীর | মাঝারি |
| Overfitting ঝুঁকি | বেশি (ছোট ডেটায়) | কম | মাঝারি |
| ছোট ডেটাসেটে পারফরম্যান্স | দুর্বল | ভালো | ভালো |
| বড় ডেটাসেটে পারফরম্যান্স | সবচেয়ে ভালো | ভালো কিন্তু ধীর | ভালো |

**সহজ সিদ্ধান্ত-নেওয়ার নিয়ম:**
- প্রচুর categorical column + ছোট/মাঝারি ডেটাসেট → **CatBoost**
- বিশাল ডেটাসেট, স্পিড সবচেয়ে জরুরি, সব সংখ্যাভিত্তিক ডেটা → **LightGBM**
- সাধারণ ব্যবহার, ভালোভাবে ডকুমেন্টেড, স্ট্যাবল সমাধান চাই → **XGBoost**

---

## অংশ ৪: Hyperparameter Tuning — প্র্যাকটিক্যাল টিপস

### ৪.১ LightGBM Tuning-এর সহজ ধাপ

1. প্রথমে `num_leaves` ঠিক করুন (সাধারণত 20-100-এর মধ্যে শুরু করুন)
2. `min_data_in_leaf` বাড়ান যদি validation score খারাপ হতে দেখেন (overfit)
3. `learning_rate` কমান, `num_boost_round` বাড়ান, `early_stopping` ব্যবহার করুন
4. `feature_fraction`, `bagging_fraction` (0.7-0.9) দিয়ে randomness যোগ করুন
5. দরকার হলে `lambda_l1`, `lambda_l2` regularization যোগ করুন

### ৪.২ CatBoost Tuning-এর সহজ ধাপ

1. `depth` (4-8) আর `learning_rate` একসাথে টিউন করুন
2. Overfit দেখলে `l2_leaf_reg` বাড়ান
3. High-cardinality categorical column থাকলে `one_hot_max_size` ঠিক করুন
4. ডেটাসেট ছোট (< ৫০ হাজার row) হলে `boosting_type='Ordered'` রাখুন
5. `bagging_temperature` দিয়ে randomness নিয়ন্ত্রণ করুন

### ৪.৩ Optuna দিয়ে অটোমেটেড Tuning

```python
import optuna
from lightgbm import LGBMClassifier
from sklearn.model_selection import cross_val_score

def objective(trial):
    params = {
        'num_leaves': trial.suggest_int('num_leaves', 20, 150),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 5, 100),
        'feature_fraction': trial.suggest_float('feature_fraction', 0.5, 1.0),
    }
    clf = LGBMClassifier(**params, n_estimators=300)
    return cross_val_score(clf, X_train, y_train, cv=5, scoring='roc_auc').mean()

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=50)
print(study.best_params)
```

---

## অংশ ৫: Feature Importance ও Interpretability

```python
# LightGBM (LGBMClassifier অবজেক্ট)
lgb.plot_importance(clf, max_num_features=15)

# CatBoost (CatBoostClassifier অবজেক্ট)
clf.get_feature_importance(prettified=True)

# SHAP — দুটোতেই কাজ করে
import shap
explainer = shap.TreeExplainer(clf)
shap_values = explainer.shap_values(X_val)
shap.summary_plot(shap_values, X_val)
```

**SHAP-এর সহজ ব্যাখ্যা**: প্রতিটা feature-এর "সত্যিকারের অবদান" মাপতে, SHAP কল্পনা করে দেখে — যদি এই feature-টা না থাকত, prediction কতটা পাল্টে যেত? এটা করা হয় **সব সম্ভাব্য combination** feature-এর জন্য, তারপর একটা fair গড় বের করা হয় (Game Theory-র Shapley Value থেকে ধার করা ধারণা):

$$\phi_i = \sum_{S \subseteq F \setminus \{i\}} \frac{|S|!(|F|-|S|-1)!}{|F|!} \left[f(S \cup \{i\}) - f(S)\right]$$

---

## অংশ ৬: সংক্ষিপ্ত সারাংশ (Cheat Sheet)

- **LightGBM** = Leaf-wise growth (সবচেয়ে লাভজনক জায়গায় split) + Histogram-based split (bucket করে দ্রুত হিসাব) + GOSS (গুরুত্বপূর্ণ ডেটায় বেশি ফোকাস) + EFB (sparse feature একত্র করা) → **সবচেয়ে দ্রুত**।
- **CatBoost** = Ordered Target Statistics (categorical encode করার সময় leakage বন্ধ) + Ordered Boosting (gradient হিসাবেও leakage বন্ধ) + Symmetric Trees (regularization + দ্রুত inference) → **categorical ডেটা ও overfitting-এ সেরা**।
- দুটোই XGBoost-এর মূল গণিত (Gradient + Hessian, Gain formula, leaf weight) শেয়ার করে — পার্থক্য শুধু tree কীভাবে বানানো হয় এবং ডেটা কীভাবে প্রসেস করা হয়, সেই ইঞ্জিনিয়ারিং-এ।

---

চাইলে আমি একটা real dataset (categorical + numerical মিশ্রিত) দিয়ে LightGBM আর CatBoost-এর end-to-end পারফরম্যান্স তুলনা করে একটা প্রজেক্ট বানিয়ে দিতে পারি, অথবা GOSS/Ordered Boosting-এর আরও গভীর step-by-step numerical উদাহরণ দেখাতে পারি।
