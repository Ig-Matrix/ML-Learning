# Olympic Medal Prediction

A beginner machine learning project that predicts how many medals a country will win at the Olympics, using a linear regression model trained on historical team data.

I built this by following the Dataquest [beginner ML project walkthrough](https://github.com/dataquestio/project-walkthroughs/tree/master/beginner_ml), typing the code myself in Jupyter notebooks inside VS Code and writing my own notes as I went. After finishing the walkthrough, I ran my own experiment to see whether more features would improve the model (see Version 2).

## The question

Can we predict how many medals a team will win at a given Games using only past data?

## Dataset

`teams.csv` has one row per team per Games. The columns I used:

| Column | Meaning |
|---|---|
| `team` | Team code, e.g. USA, NGR |
| `country` | Country name |
| `year` | Year of the Games |
| `events` | Number of different events the team entered |
| `athletes` | Number of athlete entries the team had |
| `age` | Average age of the team's athletes |
| `height` | Average height of the team's athletes |
| `weight` | Average weight of the team's athletes |
| `prev_medals` | Medals the team won at the previous Games |
| `prev_3_medals` | Average medals over the previous 3 Games |
| `medals` | Medals won (the target) |

`teams.csv` is the cleaned team-level version of the Olympic athlete events data, taken from the Dataquest walkthrough repo. The raw `athlete_events.csv` is not included here because of its size.

One thing to know about the data: medals are counted per athlete entry, so a team sport like basketball adds one medal for every player on the winning roster. That is why some big teams show counts well above 100.

## Notebooks

| Notebook | What it covers |
|---|---|
| `practice.ipynb` | Version 1: the walkthrough, with two predictors |
| `more_features.ipynb` | Version 2: testing which predictors help |

## Version 1: baseline model

1. **Explore the data.** I checked which columns move together with `medals` using a correlation matrix.
2. **Clean it.** I dropped rows with missing values. These are each team's first appearance, which has no previous Games to look back at.
3. **Split by time.** Training data is every Games before 2012 (1,609 rows). Test data is 2012 onward (405 rows). Splitting by year means the model never sees the future.
4. **Train.** A scikit-learn `LinearRegression` with two predictors: `athletes` and `prev_medals`.
5. **Predict and clean up.** Negative predictions were set to 0 and the rest rounded to whole medals.
6. **Evaluate.** I used mean absolute error, then looked at the error per team and compared it to each team's average medals.

### Findings

- `prev_medals` (0.92) and `athletes` (0.84) are strongly correlated with `medals`. `age` and `year` are close to 0.
- `athletes` and `prev_medals` are also correlated with each other (0.81), so they carry some of the same information.
- Medal counts are heavily right-skewed. Most team entries won only a few medals and a small number of teams won a lot.
- Learned equation: `medals = 0.07 * athletes + 0.75 * prev_medals - 2.142`
- The model's mean absolute error (MAE) on the test set is about **3.3 medals**, meaning predictions are off by roughly 3 medals on average.
- The error ratio (error divided by the team's average medals) is low for teams that win plenty of medals, such as France, Canada and New Zealand. It is much higher for teams that win very few, where a small miss looks big in proportion.

## Version 2: do more features help?

I kept the same train/test split and the same clean-up (negatives set to 0, then rounded), and trained a fresh linear regression for each set of predictors. Lower MAE is better.

| Predictors | MAE |
|---|---|
| events, athletes, prev_medals | 3.269 |
| athletes, prev_medals (baseline) | 3.299 |
| events, athletes, prev_medals, prev_3_medals | 3.402 |
| events, athletes, height, prev_medals, prev_3_medals | 3.442 |
| athletes, prev_medals, prev_3_medals | 3.583 |

### What I learned

- Adding `events` improved MAE by only 0.03. On 405 test rows that is close to a tie, so I treat it as "events barely helps".
- Every set that includes `prev_3_medals` scored worse. It overlaps heavily with `prev_medals`, so it adds noise more than information.
- Adding `height` also made things worse. It has a weak correlation with medals (0.13).
- More features did not mean a better model here. The simple two-feature model was already hard to beat.

### Limitations

- I picked the best set by looking at the test results, so the winning score is slightly optimistic. A separate validation set or cross-validation would be the proper way to choose.
- The test set is small, and a few large teams dominate the average error.
- Linear regression can predict negative medals, which I fix by clipping to 0.

## Next steps

- Go back to the athlete-level data and build new team features, such as number of sports entered, share of women on the team, and share of athletes who also competed at the previous Games (counting only earlier Games to avoid leaking the answer).
- Try a different model, such as a random forest, and compare it against the same baseline.

## How to run it

```bash
git clone https://github.com/ig-matrix/ml-learning.git
cd ml-learning/olympic_medal_prediction

python -m venv ml-env
source ml-env/Scripts/activate      # Git Bash on Windows
# source ml-env/bin/activate        # Mac/Linux

pip install -r requirements.txt
```

Open a notebook in VS Code and select the `ml-env` kernel, or run `jupyter notebook` after `pip install notebook`. Keep `teams.csv` in the same folder as the notebooks.

## Tools

Python, pandas, NumPy, scikit-learn, seaborn, matplotlib, Jupyter

## Credits

Project structure and dataset from the [Dataquest project walkthroughs](https://github.com/dataquestio/project-walkthroughs).
