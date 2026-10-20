# 🍛 Reusable Prompt: 7-Day South Indian Weight-Loss Meal Plan

A structured, reusable prompt that turns any AI assistant into a **senior clinical dietician** and produces a
**7-day Tamil Nadu–style meal plan**, with **Vegetarian and Non-Vegetarian options side by side**, for steady,
safe weight loss.

You edit one block (the USER PROFILE), paste the whole prompt into the assistant, and get a consistently
formatted plan every time.

---

## ✨ What you get

- Profile summary and a daily calorie target with macro split
- **Seven daily tables** (Breakfast, Lunch, Evening Snack, Dinner and Day Total) with:
  `Meal | Vegetarian | Non-Vegetarian | Portion Size | Approx. kcal`
- Authentic dish names (Ragi Dosai, Keerai Masiyal, Meen Kuzhambu …) in household portions
- Weekly grocery highlights, grouped by category
- 5–6 simple lifestyle tips and a safety disclaimer

---

## 🚀 How to use

1. Open `meal_plan_prompt.md`.
2. Edit **only** the values in the `USER PROFILE` block.
3. Copy the **entire** file and paste it into your AI assistant.
4. Review the plan; to change it, edit the profile and run again.

### Example profile edits

```
Diet type            : Vegetarian only
Allergies            : Peanuts
Foods to exclude     : Beef, Pork, Prawns
Medical conditions   : Type 2 diabetes (doctor-supervised)
Cooking time limit   : All meals ≤ 20 min
```

The prompt tells the model to replace excluded items quietly with local alternatives, so they never appear
in the plan.

---

## 🧱 Prompt structure

The prompt follows a role → context → task → guardrails → examples → format pattern:

| Section | Purpose |
|---|---|
| ✏️ **USER PROFILE** | The only part you edit: age, activity, goal, diet type, exclusions, region, cooking time |
| 1. **Role** | Senior clinical dietician specialising in Tamil Nadu home cooking |
| 2. **Context** | Sedentary professional; locally available millets, dals, greens, fish, etc. |
| 3. **Task** | 5 ordered steps: read profile → set target → build plan → **self-check** → output |
| 4. **Guardrails** | 9 rules: calorie floors, balanced meals, oil limit, real dish names, no imported "superfoods", safety disclaimer |
| 5. **Few-shot examples** | One example day, a correct substitution, and an example of what *not* to do |
| 6. **Response format** | Exact headings, table columns and section order |
| 7. **Simple tips** | The lifestyle tips to include at the end |

### Built-in quality checks (Step 4)

Before answering, the model is asked to verify that:

- no excluded or allergenic food appears
- every lunch and dinner has carbohydrate, protein and a vegetable
- no dish repeats more than twice
- millets appear at least 4 times
- portions are realistic
- daily totals fall within the target range

---

## 🛡️ Safety guardrails

- Calorie floors: never below **~1,200 kcal/day (women)** or **~1,500 kcal/day (men)**
- No fasting days, skipped meals, detox drinks or zero-carb days
- Oil limited to about 3–4 tsp per person per day; fried food at most once a week
- Always ends with a disclaimer to consult a doctor for diabetes, thyroid, kidney or heart conditions, pregnancy, or medication

---

## ⚠️ Disclaimer

This prompt produces **general guidance**, not medical advice. AI-generated calorie figures are
approximate. Anyone with a medical condition, who is pregnant, or who is on medication should check the
plan with a doctor or registered dietician before following it.

---

## 💡 Ideas to extend

- Add `Weekly budget (₹)` or `Family size` fields to the profile
- Request a Tamil-language version of the tables
- Add a "Day 8+ swap list" for variety
- Wrap it in a small Streamlit/LangChain app that fills the profile from a form

---

## 📁 Project layout

```
.
├── meal_plan_prompt.md   # the reusable prompt
├── requirements.txt      # what you need (no packages)
└── README.md
```
