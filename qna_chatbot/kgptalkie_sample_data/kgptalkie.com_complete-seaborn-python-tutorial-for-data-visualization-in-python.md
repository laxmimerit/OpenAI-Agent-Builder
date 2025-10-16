https://kgptalkie.com/complete-seaborn-python-tutorial-for-data-visualization-in-python

# Complete Seaborn Python Tutorial for Data Visualization in Python

**Source:** https://kgptalkie.com/complete-seaborn-python-tutorial-for-data-visualization-in-python

Published by georgiannacambel on 26 August 2020

## Visualizing statistical relationships

Seaborn is a Python data visualization library based on matplotlib. It provides a high-level interface for drawing attractive and informative statistical graphics.

Statistical analysis is a process of understanding how variables in a dataset relate to each other and how those relationships depend on other variables. Visualization can be a core component of this process because, when data are visualized properly, the human visual system can see trends and patterns that indicate a relationship.

---

## 1. Numerical Data Plotting

- `relplot()`
- `scatterplot()`
- `lineplot()`

---

## 2. Categorical Data Plotting

- `catplot()`
- `boxplot()`
- `stripplot()`
- `swarmplot()`
- etc…

---

## 3. Visualizing Distribution of the Data

- `distplot()`
- `kdeplot()`
- `jointplot()`
- `rugplot()`

---

## 4. Linear Regression and Relationship

- `regplot()`
- `lmplot()`

---

## 5. Controlling Plotted Figure Aesthetics

- figure styling
- axes styling
- color palettes
- etc..

---

## Importing Libraries

```python
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
```

---

## Setting Style

```python
sns.set(style='darkgrid')
```

---

## Loading Dataset

```python
tips = sns.load_dataset('tips')
tips.tail()
```

| total_bill | tip | sex | smoker | day | time | size |
|------------|-----|-----|--------|-----|------|------|
| 29.03      | 5.92| Male| No     | Sat | Dinner| 3    |
| 27.18      | 2.00| Female| Yes   | Sat | Dinner| 2    |
| 22.67      | 2.00| Male| Yes   | Sat | Dinner| 2    |
| 17.82      | 1.75| Male| No     | Sat | Dinner| 2    |
| 18.78      | 3.00| Female| No   | Thur| Dinner| 2    |

---

## Relational Plot

```python
sns.relplot(x='total_bill', y='tip', data=tips)
```

---

## Customizing Relational Plots

```python
sns.relplot(x='total_bill', y='tip', data=tips, hue='smoker', style='time')
sns.relplot(x='total_bill', y='tip', data=tips, size='size')
sns.relplot(x='total_bill', y='tip', data=tips, size='size', sizes=(15, 200))
```

---

## Line Plot

```python
df = pd.DataFrame(dict(time=np.arange(500), value=randn(500).cumsum()))
sns.relplot(x='time', y='value', kind='line', data=df, sort=True)
```

---

## FMRI Dataset

```python
fmri = sns.load_dataset('fmri')
sns.relplot(x='timepoint', y='signal', kind='line', data=fmri, ci=False)
sns.relplot(x='timepoint', y='signal', kind='line', data=fmri, ci='sd')
sns.relplot(x='timepoint', y='signal', estimator=None, kind='line', data=fmri)
```

---

## Dots Dataset

```python
dots = sns.load_dataset('dots').query("align == 'dots'")
sns.relplot(x='time', y='firing_rate', data=dots, kind='line', hue='coherence', style='choice')
```

---

## FacetGrid Examples

```python
sns.relplot(x='total_bill', y='tip', hue='smoker', col='time', data=tips)
sns.relplot(x='total_bill', y='tip', hue='smoker', col='size', data=tips)
sns.relplot(x='timepoint', y='signal', hue='subject', col='region', row='event', height=3, kind='line', estimator=None, data=fmri)
```

---

## Scatter Plots

```python
sns.scatterplot(x='total_bill', y='tip', data=tips, hue='smoker', size='size', style='time')
```

---

## Iris Dataset

```python
iris = sns.load_dataset('iris')
sns.scatterplot(x='sepal_length', y='petal_length', data=iris)
```

---

## Categorical Data Plotting

### Boxplot

```python
sns.catplot(x='day', y='total_bill', kind='box', data=tips, hue='sex')
sns.catplot(x='day', y='total_bill', kind='box', data=tips, hue='sex', dodge=False)
```

### Boxenplot

```python
diamonds = sns.load_dataset('diamonds')
sns.catplot(x='color', y='price', kind='boxen', data=diamonds.sort_values('color'))
sns.catplot(x='day', y='total_bill', kind='boxen', data=tips, dodge=False)
```

### Violin Plot

```python
sns.catplot(x='total_bill', y='day', hue='sex', kind='violin', data=tips, split=True)
```

### Swarm Plot

```python
g = sns.catplot(x='day', y='total_bill', kind='violin', inner=None, data=tips)
sns.swarmplot(x='day', y='total_bill', color='k', size=3, data=tips, ax=g.ax)
```

---

## Titanic Dataset

```python
titanic = sns.load_dataset('titanic')
sns.catplot(x='sex', y='survived', hue='class', kind='bar', data=titanic)
sns.catplot(x='deck', kind='count', palette='ch:0.95', data=titanic, hue='class')
sns.catplot(x='sex', y='survived', hue='class', kind='point', data=titanic)
```

---

## Distribution Plots

### Distplot

```python
x = np.random.randn(100)
sns.distplot(x, kde=True, hist=False, rug=False, bins=30)
```

### KDE Plot

```python
sns.kdeplot(x, shade=True, cbar=True, bw=1, cut=0)
```

### Jointplot

```python
x = tips['total_bill']
y = tips['tip']
sns.jointplot(x=x, y=y)
sns.jointplot(x=x, y=y, kind='hex')
sns.jointplot(x=x, y=y, kind='kde')
```

### Pairplot

```python
sns.pairplot(iris)
g = sns.PairGrid(iris)
g.map_diag(sns.kdeplot)
g.map_offdiag(sns.kdeplot, n_levels=10)
```

---

## Linear Regression and Relationship

### Regplot

```python
sns.regplot(x='total_bill', y='tip', data=tips)
```

### Lmplot

```python
sns.lmplot(x='total_bill', y='tip', data=tips)
sns.lmplot(x='size', y='tip', data=tips, x_jitter=0.05)
sns.lmplot(x='size', y='tip', data=tips, x_estimator=np.mean)
```

### Anscombe Dataset

```python
data = sns.load_dataset('anscombe')
sns.lmplot(x='x', y='y', data=data.query("dataset == 'I'"), ci=None, scatter_kws={'s': 80})
sns.lmplot(x='x', y='y', data=data.query("dataset == 'II'"), ci=None, scatter_kws={'s': 80}, order=2)
sns.lmplot(x='x', y='y', data=data.query("dataset == 'III'"), ci=None, scatter_kws={'s': 80}, robust=True)
```

---

## Controlling Figure Aesthetics

### Styling

```python
def sinplot(flip=1):
    x = np.linspace(0, 14, 100)
    for i in range(1, 7):
        plt.plot(x, np.sin(x + i * 0.5) * (7 - i) * flip)

sns.set_style('ticks', {'axes.grid': True, 'xtick.direction': 'in'})
sinplot()
sns.despine(left=True, bottom=False)
```

### Color Palettes

```python
current_palettes = sns.color_palette()
sns.palplot(current_palettes)
sns.palplot(sns.color_palette('hls', 8))
```

---

## Conclusion

With the help of data visualization, we can see how the data looks like and what kind of correlation is held by the attributes of data.

This is the first and foremost step where they will get a high-level statistical overview on how the data is and some of its attributes like the underlying distribution, presence of outliers, and several more useful features.

From the perspective of building models, by visualizing the data we can find the hidden patterns, explore if there are any clusters within data and we can find if they are linearly separable/too much overlapped etc.

From this initial analysis we can easily rule out the models that won’t be suitable for such a data and we will implement only the models that are suitable, without wasting our valuable time and the computational resources.

**Source:** https://kgptalkie.com/complete-seaborn-python-tutorial-for-data-visualization-in-python