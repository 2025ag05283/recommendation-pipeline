# Feature Catalog

Feature version **1.0.0** - all features are computed **as of the train cutoff** from training data only (no test leakage). Stored in `data/features/` and served to training and prediction through `src/features/store.py::FeatureStore`.

## User features (key: `user_id`) - 30 features

| Feature | Type | Description | Computation | Used by | Null % | Statistics |
|---|---|---|---|---|---:|---|
| `user_age` | int | Age of the user | users.age (validated) | profile, hybrid | 0.0 | min=7, mean=34.1, max=73 |
| `user_gender` | category | Gender code (M/F) | users.gender upper-cased | profile | 0.0 | 2 values, top='M' |
| `user_occupation` | category | Occupation (missing -> 'unknown') | users.occupation cleaned | profile | 0.0 | 22 values, top='student' |
| `user_age_group` | category | Age bucket | pd.cut(age, [0,17,24,34,44,54,120]) | profile | 0.0 | 6 values, top='25-34' |
| `user_activity_count` | int | Number of training interactions | count(train rows per user) | hybrid, analysis | 0.0 | min=0, mean=84.8, max=737 |
| `user_avg_rating` | float | Average rating given by the user | mean(train.rating per user) | hybrid, MF diagnostics | 0.0 | min=1.49, mean=3.57, max=4.87 |
| `user_rating_std` | float | Std. dev. of the user's ratings (0 if one rating) | std(train.rating per user) | analysis | 0.0 | min=0, mean=0.807, max=1.75 |
| `user_days_since_last_interaction` | float | Days between the user's last training interaction and the reference date (train cutoff) | (cutoff - max(train.datetime)) in days | recency, analysis | 0.0 | min=-1, mean=59, max=167 |
| `user_active_span_days` | float | Days between first and last training interaction | max(datetime) - min(datetime) | analysis | 0.0 | min=0, mean=16, max=167 |
| `user_preferred_genre` | category | Genre the user rated most often among liked items (rating >= 4); falls back to all items | argmax over genre counts of liked items | explanations, hybrid | 0.0 | 10 values, top='drama' |
| `user_n_distinct_genres` | int | Number of distinct genres the user interacted with | count(genres with > 0 interactions) | diversity analysis | 0.0 | min=0, mean=12.1, max=19 |
| `user_genre_pref_unknown` | float | Share of the user's liked items that belong to genre 'unknown' | sum(liked items with genre) / sum(all genre flags of liked items) | hybrid (content affinity) | 0.0 | min=0, mean=7.46e-05, max=0.05 |
| `user_genre_pref_action` | float | Share of the user's liked items that belong to genre 'action' | sum(liked items with genre) / sum(all genre flags of liked items) | hybrid (content affinity) | 0.0 | min=0, mean=0.0896, max=0.357 |
| `user_genre_pref_adventure` | float | Share of the user's liked items that belong to genre 'adventure' | sum(liked items with genre) / sum(all genre flags of liked items) | hybrid (content affinity) | 0.0 | min=0, mean=0.0453, max=0.212 |
| `user_genre_pref_animation` | float | Share of the user's liked items that belong to genre 'animation' | sum(liked items with genre) / sum(all genre flags of liked items) | hybrid (content affinity) | 0.0 | min=0, mean=0.0113, max=0.111 |
| `user_genre_pref_childrens` | float | Share of the user's liked items that belong to genre 'childrens' | sum(liked items with genre) / sum(all genre flags of liked items) | hybrid (content affinity) | 0.0 | min=0, mean=0.019, max=0.154 |
| `user_genre_pref_comedy` | float | Share of the user's liked items that belong to genre 'comedy' | sum(liked items with genre) / sum(all genre flags of liked items) | hybrid (content affinity) | 0.0 | min=0, mean=0.0953, max=1 |
| `user_genre_pref_crime` | float | Share of the user's liked items that belong to genre 'crime' | sum(liked items with genre) / sum(all genre flags of liked items) | hybrid (content affinity) | 0.0 | min=0, mean=0.0351, max=0.156 |
| `user_genre_pref_documentary` | float | Share of the user's liked items that belong to genre 'documentary' | sum(liked items with genre) / sum(all genre flags of liked items) | hybrid (content affinity) | 0.0 | min=0, mean=0.00315, max=0.111 |
| `user_genre_pref_drama` | float | Share of the user's liked items that belong to genre 'drama' | sum(liked items with genre) / sum(all genre flags of liked items) | hybrid (content affinity) | 0.0 | min=0, mean=0.17, max=0.52 |
| `user_genre_pref_fantasy` | float | Share of the user's liked items that belong to genre 'fantasy' | sum(liked items with genre) / sum(all genre flags of liked items) | hybrid (content affinity) | 0.0 | min=0, mean=0.00296, max=0.0588 |
| `user_genre_pref_film_noir` | float | Share of the user's liked items that belong to genre 'film_noir' | sum(liked items with genre) / sum(all genre flags of liked items) | hybrid (content affinity) | 0.0 | min=0, mean=0.00802, max=0.0938 |
| `user_genre_pref_horror` | float | Share of the user's liked items that belong to genre 'horror' | sum(liked items with genre) / sum(all genre flags of liked items) | hybrid (content affinity) | 0.0 | min=0, mean=0.0168, max=0.418 |
| `user_genre_pref_musical` | float | Share of the user's liked items that belong to genre 'musical' | sum(liked items with genre) / sum(all genre flags of liked items) | hybrid (content affinity) | 0.0 | min=0, mean=0.0139, max=0.108 |
| `user_genre_pref_mystery` | float | Share of the user's liked items that belong to genre 'mystery' | sum(liked items with genre) / sum(all genre flags of liked items) | hybrid (content affinity) | 0.0 | min=0, mean=0.0241, max=0.333 |
| `user_genre_pref_romance` | float | Share of the user's liked items that belong to genre 'romance' | sum(liked items with genre) / sum(all genre flags of liked items) | hybrid (content affinity) | 0.0 | min=0, mean=0.0805, max=0.444 |
| `user_genre_pref_sci_fi` | float | Share of the user's liked items that belong to genre 'sci_fi' | sum(liked items with genre) / sum(all genre flags of liked items) | hybrid (content affinity) | 0.0 | min=0, mean=0.0476, max=0.257 |
| `user_genre_pref_thriller` | float | Share of the user's liked items that belong to genre 'thriller' | sum(liked items with genre) / sum(all genre flags of liked items) | hybrid (content affinity) | 0.0 | min=0, mean=0.085, max=0.391 |
| `user_genre_pref_war` | float | Share of the user's liked items that belong to genre 'war' | sum(liked items with genre) / sum(all genre flags of liked items) | hybrid (content affinity) | 0.0 | min=0, mean=0.0433, max=0.194 |
| `user_genre_pref_western` | float | Share of the user's liked items that belong to genre 'western' | sum(liked items with genre) / sum(all genre flags of liked items) | hybrid (content affinity) | 0.0 | min=0, mean=0.00509, max=0.0769 |

## Item features (key: `item_id`) - 29 features

| Feature | Type | Description | Computation | Used by | Null % | Statistics |
|---|---|---|---|---|---:|---|
| `item_popularity` | int | Number of training ratings of the item | count(train rows per item) | popularity, hybrid | 0.0 | min=0, mean=47.6, max=473 |
| `item_popularity_rank` | int | Rank by popularity (1 = most popular) | rank(item_popularity, desc) | analysis | 0.0 | min=1, mean=825, max=1.62e+03 |
| `item_user_share` | float | Fraction of all users who rated the item | item_popularity / n_users | novelty | 0.0 | min=0, mean=0.0504, max=0.502 |
| `item_avg_rating` | float | Mean training rating of the item (NaN-safe: global mean if unrated) | mean(train.rating per item) | analysis | 0.0 | min=1, mean=3.06, max=5 |
| `item_bayesian_avg_rating` | float | Bayesian-average rating shrunk towards the global mean | (m*mu + sum r)/(m + n), m = bayesian_prior_weight | popularity rating prediction | 0.0 | min=2.08, mean=3.36, max=4.44 |
| `item_rating_std` | float | Std. dev. of the item's ratings | std(train.rating per item) | analysis | 0.0 | min=0, mean=0.926, max=2.83 |
| `item_release_year` | int | Release year (median-imputed when missing) | year(release_date) | content | 0.0 | min=1.92e+03, mean=1.99e+03, max=2e+03 |
| `item_age_years` | float | Item age at the reference date | (cutoff - release_date) / 365.25 | content | 0.0 | min=-0.63, mean=8.61, max=76.2 |
| `item_days_since_last_interaction` | float | Days since the item was last rated (NaN-safe: large value if never) | (cutoff - max(train.datetime per item)) in days | trend analysis | 0.0 | min=0, mean=409, max=1e+04 |
| `item_n_genres` | int | Number of genres of the item | sum(genre flags) | content | 0.0 | min=1, mean=1.72, max=6 |
| `item_genre_unknown` | int | 1 if the item belongs to genre 'unknown' | u.item genre flag 'unknown' | hybrid (content), diversity | 0.0 | min=0, mean=0.00119, max=1 |
| `item_genre_action` | int | 1 if the item belongs to genre 'action' | u.item genre flag 'action' | hybrid (content), diversity | 0.0 | min=0, mean=0.149, max=1 |
| `item_genre_adventure` | int | 1 if the item belongs to genre 'adventure' | u.item genre flag 'adventure' | hybrid (content), diversity | 0.0 | min=0, mean=0.0803, max=1 |
| `item_genre_animation` | int | 1 if the item belongs to genre 'animation' | u.item genre flag 'animation' | hybrid (content), diversity | 0.0 | min=0, mean=0.025, max=1 |
| `item_genre_childrens` | int | 1 if the item belongs to genre 'childrens' | u.item genre flag 'childrens' | hybrid (content), diversity | 0.0 | min=0, mean=0.0725, max=1 |
| `item_genre_comedy` | int | 1 if the item belongs to genre 'comedy' | u.item genre flag 'comedy' | hybrid (content), diversity | 0.0 | min=0, mean=0.3, max=1 |
| `item_genre_crime` | int | 1 if the item belongs to genre 'crime' | u.item genre flag 'crime' | hybrid (content), diversity | 0.0 | min=0, mean=0.0648, max=1 |
| `item_genre_documentary` | int | 1 if the item belongs to genre 'documentary' | u.item genre flag 'documentary' | hybrid (content), diversity | 0.0 | min=0, mean=0.0297, max=1 |
| `item_genre_drama` | int | 1 if the item belongs to genre 'drama' | u.item genre flag 'drama' | hybrid (content), diversity | 0.0 | min=0, mean=0.431, max=1 |
| `item_genre_fantasy` | int | 1 if the item belongs to genre 'fantasy' | u.item genre flag 'fantasy' | hybrid (content), diversity | 0.0 | min=0, mean=0.0131, max=1 |
| `item_genre_film_noir` | int | 1 if the item belongs to genre 'film_noir' | u.item genre flag 'film_noir' | hybrid (content), diversity | 0.0 | min=0, mean=0.0143, max=1 |
| `item_genre_horror` | int | 1 if the item belongs to genre 'horror' | u.item genre flag 'horror' | hybrid (content), diversity | 0.0 | min=0, mean=0.0547, max=1 |
| `item_genre_musical` | int | 1 if the item belongs to genre 'musical' | u.item genre flag 'musical' | hybrid (content), diversity | 0.0 | min=0, mean=0.0333, max=1 |
| `item_genre_mystery` | int | 1 if the item belongs to genre 'mystery' | u.item genre flag 'mystery' | hybrid (content), diversity | 0.0 | min=0, mean=0.0363, max=1 |
| `item_genre_romance` | int | 1 if the item belongs to genre 'romance' | u.item genre flag 'romance' | hybrid (content), diversity | 0.0 | min=0, mean=0.147, max=1 |
| `item_genre_sci_fi` | int | 1 if the item belongs to genre 'sci_fi' | u.item genre flag 'sci_fi' | hybrid (content), diversity | 0.0 | min=0, mean=0.06, max=1 |
| `item_genre_thriller` | int | 1 if the item belongs to genre 'thriller' | u.item genre flag 'thriller' | hybrid (content), diversity | 0.0 | min=0, mean=0.149, max=1 |
| `item_genre_war` | int | 1 if the item belongs to genre 'war' | u.item genre flag 'war' | hybrid (content), diversity | 0.0 | min=0, mean=0.0422, max=1 |
| `item_genre_western` | int | 1 if the item belongs to genre 'western' | u.item genre flag 'western' | hybrid (content), diversity | 0.0 | min=0, mean=0.0161, max=1 |

## Interaction features (key: `['user_id', 'item_id']`) - 6 features

| Feature | Type | Description | Computation | Used by | Null % | Statistics |
|---|---|---|---|---|---:|---|
| `rating` | int | Explicit rating (label) | train.rating | MF training target | 0.0 | min=1, mean=3.52, max=5 |
| `is_relevant` | int | 1 if rating >= relevance_threshold | rating >= 4 | ranking labels | 0.0 | min=0, mean=0.551, max=1 |
| `rating_dev_from_user_avg` | float | Rating minus the user's average rating | rating - user_avg_rating | analysis, bias modelling | 0.0 | min=-3.66, mean=-2.55e-18, max=3.51 |
| `rating_dev_from_item_avg` | float | Rating minus the item's average rating | rating - item_avg_rating | analysis, bias modelling | 0.0 | min=-3.54, mean=9.95e-18, max=3.17 |
| `interaction_recency_days` | float | Days between the interaction and the reference date | (cutoff - datetime) in days | recency weighting | 0.0 | min=0, mean=88.8, max=168 |
| `user_interaction_seq` | int | Order of the interaction in the user's history (1 = first) | cumcount over user, sorted by time | sequence analysis | 0.0 | min=1, mean=100, max=737 |
