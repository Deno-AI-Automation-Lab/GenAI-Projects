# Knowledge-graph examples (hybrid vs hybrid + graph)

Of 30 golden questions, adding the Neo4j graph ranked the correct section higher in **4**, lower in 4 and the same in 22 (top-5, re-ranking off).

## Example 1: What should I do if I forgot my password and can't log in to my account?
- Product: `aerobeat-x9-wireless-noise-cancelling-headphones`
- Correct section: account_and_login.docx › 2. Forgot Password - Exact Reset Steps
- Rank without graph: **3** → with graph: **1**
- Found by: bm25, entity_summary, graph, vector
- Graph facts used: account -can be created using-> verified email; account -is not locked-> caps lock; account -causes-> failed password attempts; account -are rejected-> valid credentials; account -can open-> customers

## Example 2: What should I do if I received a login code that I didn't request?
- Product: `puresweep-v12-robot-vacuum-mop`
- Correct section: account_and_login.docx › 10. Login Scenarios and Support Escalation
- Rank without graph: **2** → with graph: **1**
- Found by: entity_summary, graph, vector
- Graph facts used: login code -triggers-> password reset; login code -should not be-> shared; login code -triggers-> account security review; code -should be added before-> dispatch; code -requires-> secure package room

## Example 3: Under what circumstances can I get a replacement for my lost package?
- Product: `smartcrisp-pro-8l-wi-fi-air-fryer`
- Correct section: shipping_policy.docx › When a replacement may be approved
- Rank without graph: **2** → with graph: **1**
- Found by: entity_summary, graph, vector
- Graph facts used: replacement -issued by-> kitchennova; replacement -or offers-> refund; replacement -requested after-> verification; replacement -ships after-> carrier scan; replacement -depends on-> stock

## Example 4: What should I do if my package hasn't been updated in tracking for several days?
- Product: `smartcrisp-pro-8l-wi-fi-air-fryer`
- Correct section: shipping_policy.docx › Lost in transit
- Rank without graph: **4** → with graph: **1**
- Found by: bm25, entity_summary, graph, vector
- Graph facts used: stolen packages -different rules apply to-> packages in transit; stolen packages -different rules apply to-> packages marked delivered; missing packages -different rules apply to-> packages in transit; missing packages -different rules apply to-> packages marked delivered; packages in transit -different rules apply to-> stolen packages
