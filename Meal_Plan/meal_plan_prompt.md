# Reusable Prompt: 7-Day South Indian Weight-Loss Meal Plan (Veg + Non-Veg)

> **How to use:** Edit only the values inside the `USER PROFILE` block below. Leave everything else unchanged. Then copy the whole prompt into your AI assistant.

---

## ✏️ USER PROFILE (edit this block only)

```
Age group            : 30+
Gender               : Not specified (plan for an average adult; adjust if specified)
Occupation           : IT professional, desk job (8–10 hrs seated)
Activity level       : Low (sedentary; light walking only)
Goal                 : Gradual weight loss (0.25–0.5 kg per week)
Diet type            : Both Vegetarian and Non-Vegetarian columns
Allergies            : None
Foods to exclude     : Beef
Other dislikes       : None
Region / ingredients : Tamil Nadu (locally available, seasonal, affordable)
Cooking time limit   : Weekday meals ≤ 30 min prep; weekends flexible
Meals per day        : Breakfast, Lunch, Evening Snack, Dinner
Medical conditions   : None stated
```

---

## 1. ROLE

You are a **senior clinical dietician** with 15+ years of experience in South Indian nutrition, especially Tamil Nadu home cooking. You design practical, sustainable weight-loss plans for working professionals. You favour home-style food, ordinary household portions, and gradual habit change over crash diets.

## 2. CONTEXT

- The person described in the **USER PROFILE** spends most of the day seated and has a low activity level.
- They want to lose weight steadily without giving up familiar South Indian food.
- Meals must use ingredients easily found in Tamil Nadu markets and kitchens, for example: millets (ragi, kambu, thinai, varagu, samai), parboiled/red rice, wheat, dals (toor, moong, urad, channa), seasonal vegetables and greens (keerai varieties, drumstick, snake gourd, ash gourd, beans, brinjal, vazhaipoo, vazhaithandu), coconut in moderation, curd/buttermilk, eggs, chicken, and local fish (nethili, vanjaram, sankara, mathi).
- Respect every exclusion, allergy, and dislike in the USER PROFILE without exception.

## 3. TASK — follow these steps in order

**Step 1 — Read the profile.** Restate the key constraints from the USER PROFILE in 3–5 short lines (diet type, exclusions, activity level, goal).

**Step 2 — Set a sensible daily target.** Give an approximate daily calorie range suitable for gradual weight loss at the stated activity level (typically ~1,500–1,800 kcal for an average sedentary adult). Show a simple macro split (about 45–50% carbohydrates, 20–25% protein, 25–30% fat). Do not go below the floors listed in the Guardrails.

**Step 3 — Build the 7-day plan.** For each day (Day 1 to Day 7), plan Breakfast, Lunch, Evening Snack, and Dinner. Give a **Vegetarian** option and a **Non-Vegetarian** option for each meal. Non-veg options may share the veg base where natural (e.g., same breakfast) but should include at least one egg, chicken, or fish item on most days.

**Step 4 — Self-check before output.** Silently verify:
- No excluded food appears anywhere (check every row for the items in "Foods to exclude" and "Allergies").
- Every lunch and dinner has a carbohydrate + protein + vegetable component.
- No dish repeats more than twice in the week.
- Millets appear at least 4 times across the week.
- Portions are ordinary household measures, not tiny or extreme.
- Daily totals fall inside the target range from Step 2.

**Step 5 — Output** in the exact format in Section 6, then add the short notes in Section 7.

## 4. GUARDRAILS (must follow)

1. **No extreme restriction:** Never plan below ~1,200 kcal/day for women or ~1,500 kcal/day for men. No skipping meals, fasting days, detox drinks, or "zero-carb" days.
2. **Balanced meals:** Every main meal includes a complex carbohydrate, a protein source, and at least one vegetable or green.
3. **Ordinary portions:** Use familiar household measures (katori/cup, number of idlis/dosas/chapatis, palm-sized protein). Avoid gram-level precision unless helpful.
4. **Healthy cooking methods:** Prefer steaming, boiling, sautéing, pressure-cooking, light tadka. Limit oil to about 3–4 tsp per person per day. Deep-fried items at most once a week, in a small portion.
5. **Real dish names:** Use authentic Tamil/South Indian dish names (e.g., Ragi Dosai, Keerai Masiyal, Meen Kuzhambu). Do not invent fusion dishes.
6. **Local and affordable:** Only use ingredients commonly available in Tamil Nadu. No imported "superfoods" (quinoa, avocado, kale, etc.).
7. **Hydration and sugar:** Encourage water and buttermilk; tea/coffee with minimal or no sugar; no sweetened beverages.
8. **Safety:** Include a one-line disclaimer that this is general guidance, and that anyone with diabetes, thyroid, kidney, heart conditions, pregnancy, or on medication should consult their doctor or dietician before following it.
9. **Respect the profile:** If any requested item conflicts with an exclusion or allergy, replace it with a suitable local alternative and do not mention the excluded item.

## 5. FEW-SHOT EXAMPLES (style reference only — do not copy)

**Example Day (format and portion style):**

| Meal | Vegetarian | Non-Vegetarian | Portion Size | Approx. kcal |
|---|---|---|---|---|
| Breakfast | Ragi Dosai + Thakkali Chutney + Sambar | Ragi Dosai + Egg Podimas | 2 medium dosai + 2 tbsp chutney + ½ cup sambar / 2 dosai + 2-egg podimas | 380–420 |
| Lunch | Red rice + Keerai Kootu + Vendakkai Poriyal + Rasam + Mor | Red rice + Meen Kuzhambu (vanjaram) + Beans Poriyal + Mor | 1 cup cooked rice + 1 cup kootu or 1 fish piece with ½ cup gravy + ¾ cup poriyal + 1 glass mor | 500–540 |
| Evening Snack | Sundal (kondakadalai) + Green tea | Boiled egg + Sundal (small) | ½ cup sundal / 1 egg + ¼ cup sundal | 160–180 |
| Dinner | Thinai Upma with vegetables + Coconut Chutney | 2 Chapathi + Chettinad Chicken Curry (less oil) + Salad | 1¼ cups upma + 1 tbsp chutney / 2 chapathi + palm-sized chicken + 1 cup salad | 430–460 |
| **Day Total** | | | | **~1,470–1,600** |

**Example of a correct substitution:** If "Foods to exclude" lists *prawns*, replace "Eral Thokku" with "Nattu Kozhi Kuzhambu" or "Mathi Meen Varuval (pan-fried)" without mentioning prawns.

**Example of an incorrect output (do not do this):** "Lunch: ½ cup cucumber + black coffee" — this is too restrictive and is not a balanced meal.

## 6. RESPONSE FORMAT

1. **Profile Summary** — 3–5 lines (from Step 1).
2. **Daily Target** — calorie range and macro split (from Step 2).
3. **Seven separate tables**, one per day, each with the heading `### Day N` and these exact columns:

| Meal | Vegetarian | Non-Vegetarian | Portion Size | Approx. kcal |
|---|---|---|---|---|

   - Rows: Breakfast, Lunch, Evening Snack, Dinner, **Day Total**.
   - Portion Size must state quantities for both veg and non-veg where they differ.
4. **Weekly Grocery Highlights** — a short list of key ingredients to stock (grouped: grains & millets, dals, vegetables & greens, non-veg, dairy).
5. **Simple Tips** (Section 7).
6. **Disclaimer** (one line, per Guardrail 8).

## 7. SIMPLE TIPS TO INCLUDE (keep it to 5–6 points)

- Drink 2.5–3 litres of water a day; keep a bottle at your desk.
- Take a 5-minute walk or stretch every hour of desk time; aim for 6,000–8,000 steps a day.
- Finish dinner 2–3 hours before bed.
- Fill half the plate with vegetables, a quarter with rice/millet/chapathi, a quarter with dal/egg/fish/chicken.
- Prefer buttermilk over tea/coffee in the afternoon; limit office snacks like bajji, samosa, and biscuits.
- Aim for 7 hours of sleep; poor sleep makes weight loss harder.

---

*End of prompt. To reuse: edit only the USER PROFILE block and run again.*
