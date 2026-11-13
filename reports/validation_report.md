# Data Validation Report

Generated: 2026-11-14 00:29:06  
Quality gate: max invalid-interaction rate = 5%

## 1. Row counts

| Table | Raw rows | Valid rows | Rejected rows | Rejected % |
|---|---:|---:|---:|---:|
| users | 949 | 943 | 6 | 0.632% |
| items | 1,686 | 1,682 | 4 | 0.237% |
| interactions | 100,024 | 100,000 | 24 | 0.024% |

## 2. Rule-based checks (rows failing each rule)

| Table | Check | Category | Rule | Failed rows | Status |
|---|---|---|---|---:|---|
| users | `user_required_fields` | missing | user_id, age and gender must be present | 1 | FAIL |
| users | `user_id_integer` | type | user_id must be an integer | 0 | PASS |
| users | `age_integer` | type | age must be an integer | 1 | FAIL |
| users | `age_in_range` | range | age must be within [1, 100] | 2 | FAIL |
| users | `gender_allowed` | range | gender must be one of ['F', 'M'] | 1 | FAIL |
| users | `user_id_unique` | duplicate | user_id must be unique (first occurrence kept) | 1 | FAIL |
| items | `item_required_fields` | missing | item_id and title must be present | 1 | FAIL |
| items | `item_id_integer` | type | item_id must be an integer | 0 | PASS |
| items | `release_date_format` | timestamp | release_date, if present, must be a valid DD-Mon-YYYY date | 1 | FAIL |
| items | `genre_flags_binary` | type | every genre flag must be 0 or 1 | 1 | FAIL |
| items | `item_id_unique` | duplicate | item_id must be unique (first occurrence kept) | 1 | FAIL |
| interactions | `interaction_required_fields` | missing | user_id, item_id, rating and timestamp must be present | 4 | FAIL |
| interactions | `ids_integer` | type | user_id and item_id must be integers | 1 | FAIL |
| interactions | `rating_numeric` | type | rating must be numeric | 1 | FAIL |
| interactions | `rating_valid` | range | rating must be an integer in [1, 5] | 5 | FAIL |
| interactions | `timestamp_numeric` | timestamp | timestamp must be a unix epoch integer | 1 | FAIL |
| interactions | `timestamp_plausible` | timestamp | timestamp must be between 1995-01-01 and now | 2 | FAIL |
| interactions | `no_exact_duplicates` | duplicate | exact duplicate interaction rows are not allowed (first kept) | 10 | FAIL |
| interactions | `user_exists` | referential | user_id must exist in the validated users table | 4 | FAIL |
| interactions | `item_exists` | referential | item_id must exist in the validated items table | 2 | FAIL |

## 3. Warnings - missing optional values (kept, imputed in preparation)

| Table | Column | Missing values |
|---|---|---:|
| users | occupation | 5 |
| users | zip_code | 3 |
| items | release_date | 4 |

## 4. Pandera schema validation (independent declarative layer)

Pandera available: **True**

| Table | Column | Check | Failures |
|---|---|---|---:|
| users | age | coerce_dtype('int64') | 2 |
| users | age | dtype('int64') | 2 |
| users | age | in_range(1, 100) | 1 |
| users | age | not_nullable | 1 |
| users | gender | isin(['M', 'F']) | 1 |
| users | user_id | field_uniqueness | 2 |
| items | genre_action | isin([0, 1]) | 1 |
| items | item_id | field_uniqueness | 2 |
| items | title | not_nullable | 1 |
| interactions | rating | coerce_dtype('float64') | 1 |
| interactions | rating | dtype('float64') | 1 |
| interactions | rating | isin([1.0, 2.0, 3.0, 4.0, 5.0]) | 100023 |
| interactions | rating | not_nullable | 1 |
| interactions | timestamp | coerce_dtype('int64') | 1 |
| interactions | timestamp | dtype('int64') | 1 |
| interactions | timestamp | greater_than_or_equal_to(788918400) | 1 |
| interactions | user_id | coerce_dtype('int64') | 4 |
| interactions | user_id | dtype('int64') | 4 |
| interactions | user_id | not_nullable | 3 |

## 5. Outputs

* Valid typed tables: `data/processed/users_validated`, `items_validated`, `interactions_validated`
* Error datasets (original values + `error_reasons`): `data/errors/users_errors.csv`, `items_errors.csv`, `interactions_errors.csv`, combined in `all_errors.csv`
