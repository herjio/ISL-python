
## Chapter 2: Statistical Learning

### 2.1: What is statistical learning


$Y = f(X) + \epsilon$
X is the input variable 
$\epsilon$  is the Irreducible error

Here f is some fixed but unknown function of $X1, . . . , Xp, and ϵ$ is a random error term, which is independent of X and has mean zero. In this formulation, f represents the systematic information that X provides about Y

In essence, ==statistical learning refers to a set of approaches for estimating [[f]]== 

$Y = f(X) + \epsilon$

**We almost never know what $f$ actually is.** Because it is invisible and unknown, we are forced to build models ($\hat{f}$) to try and approximate it as closely as possible.

- **$X$ (Inputs):** The time of day, the weather, and the day of the week.
    
- **$\hat{f}$ (Your Model):** The algorithm inside your mapping app that estimates the drive will take 25 minutes.
    
- **$f$ (The Truth):** The actual, literal laws of physics, city infrastructure, and human behavior that dictate exactly how traffic flows on those specific roads at that exact second.
    
- **$\epsilon$ (Irreducible Error):** The random squirrel that runs into the road.


#### 2.1.1: Why do we need to estimate f?
- Prediction 
- Inference

1. ***Prediction*** 
In many situations, a set of inputs X are readily available, but the output Y cannot be easily obtained. In this setting, since the error term averages to zero, we can predict Y using

$\hat{Y} = \hat{f}(X)$

where $\hat{f}$ represents our estimate for $f$, and $\hat{Y}$ represents the resulting prediction for Y . In this setting, $\hat{f}$ is often treated as a black box, in the sense that one is not typically concerned with the exact form of $\hat{f}$, provided that it yields accurate predictions for Y.

2. ***Inference***
This section shifts gears from **Prediction** to **Inference**.

In the previous section, the goal was just to get the right answer (Prediction). We didn't care _how_ the model made its guess, which is why we could use a "black box."

**Inference** is the exact opposite. With inference, we aren't just trying to guess the outcome ($Y$). We want to open the box, look at the gears, and understand exactly _how_ the inputs ($X$) are affecting the outcome. Because we need to see how the math works, we **cannot** use a black box.

#### 2.1.2: How do we estimate f?

Most statistical learning methods for this task can be characterized as either *parametric* or *non-parametric.*

**Parametric**
Parametric methods involve a two-step model-based approach. 
1. First, we make an assumption about the functional form, or shape, of f. For example, one very simple assumption is that f is linear in X:
$$f(X) = \beta_0 + \beta_1X_1 + \beta_2X_2 + \dots + \beta_pX_p$$
	This is a linear model. Once we have assumed that f is linear, the problem of estimating f is greatly simplified. Instead of having to estimate an entirely arbitrary p-dimensional function f(X), one only needs to estimate the p + 1 coefficients β0, β1, . . . , βp.

2. After a model has been selected, we need a procedure that uses the training data to fit or train the model. We need to estimate the parameters β0, β1, . . . , βp. That is, we want to find values of these parameters such that
$$f(X) ≈ \beta_0 + \beta_1X_1 + \beta_2X_2 + \dots + \beta_pX_p$$
	The most common approach to fitting the model is referred to as (ordinary) least squares, which we discuss in Chapter 3. However, least squares least squares is one of many possible ways to fit the linear model.

	The model-based approach just described is referred to as parametric; it reduces the problem of estimating f down to one of estimating a set of parameters. Assuming a parametric form for f simplifies the problem of estimating f because it is generally much easier to estimate a set of parameters, such as β0, β1, . . . , βp in the linear model (2.4), than it is to fit an entirely arbitrary function f.
	The potential disadvantage of a parametric approach is ==that the model we choose will usually not match the true unknown form of f.== If the chosen model is too far from the true f, then our estimate will be poor.

**Non-Parametric Methods**
Non-parametric methods do not make explicit assumptions about the functional form of f. Instead they seek an estimate of f that gets as close to the data points as possible without being too rough or wiggly.
Such approaches can have a major **advantage** over parametric approaches:
- By avoiding the assumption of a particular functional form for f, they have the potential to accurately fit a wider range of possible shapes for f.

But non-parametric approaches do suffer from a major **disadvantage**: 
- since they do not reduce the problem of estimating f to a small number of parameters, a very large number of observations (far more than is typically needed for a parametric approach) is required in order to obtain an accurate estimate for f.


#### 2.1.4 Supervised Versus Unsupervised Learning

Most statistical learning problems fall into one of two categories: supervised supervised or unsupervised.
We wish to fit a model that relates the response to the predictors, with the aim of accurately predicting the response for future observations (prediction) or ==better understanding the relationship between the response and the predictors (inference).==

It is not possible to fit a linear regression model, since there is no response variable to predict. In this setting, we are in some sense working blind; the situation is referred to as unsupervised because we lack a response variable that can supervise our analysis.

What sort of statistical analysis is possible? We can seek to understand the relationships between the variables or between the observations. One statistical learning tool that we may use in this setting is cluster analysis, or clustering.

#### 2.1.5 Regression Versus Classification Problems

Variables can be characterized as either quantitative or qualitative (also known as categorical). Quantitative variables take on numerical values.
In contrast, qualitative variables take on values in one of K different classes, or categories.

We tend to refer to problems with a quantitative response as regression problems, while those involving a qualitative response are often referred to as classification problems.

We tend to select statistical learning methods on the basis of whether the response is quantitative or qualitative; i.e. we might use linear regression when quantitative and logistic regression when qualitative.

### 2.2 Assessing Model Accuracy

#### 2.2.1 Measuring the Quality of Fit

In order to evaluate the performance of a statistical learning method on a given data set, we need some way to measure how well its predictions actually match the observed data.

In the regression setting, the most commonly-used measure is the mean squared error (MSE), given by

$$\text{MSE} = \frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{f}(x_i))^2$$

where $\hat{f}(x_i)$ is the prediction that $\hat{f}$ gives for the $i^{th}$ observation. The MSE will be small if the predicted responses are very close to the true responses, and will be large if for some of the observations, the predicted and true responses differ substantially.

But in general, we do not really care how well the method works training MSE on the training data. *Rather, we are interested in the accuracy of the predictions that we obtain when we apply our method to previously unseen test data.*

However, we are really not interested in whether $\hat{f}(x_i) ≈ y_i$ ; instead, we want to know whether $\hat{f}(x_0)$ is approximately equal to $y_0$, where $(x_0, y_0)$ is a previously unseen test observation not used to train the statistical learning method.
We want to choose the method that gives the lowest test MSE, as opposed to the lowest training MSE. In other words, if we had a large number of test observations, we could compute,

Ave$(y_0 − \hat{f}(x_0))^2$

We observe that the training MSE decreases monotonically as the model flexibility increases.
A more flexible model is avoided since it overfits the training data and fails to completely predict on new unseen data.

#### 2.2.2 The Bias-Variance Trade-Off

In order to minimize the expected test error, we need to select a statistical learning method that simultaneously achieves *low variance* and *low bias.*

**Variance** refers to the amount by which $\hat{f}$ would change if we estimated it using a different training data set.
But ideally the estimate for f should not vary too much between training sets. However, if a method has high variance then small changes in the training data can result in large changes in $\hat{f}$.
==High flexibility models have high variance==.
**High variance** (for instance, by drawing a curve that passes through every single training observation). 

**bias** refers to the error that is introduced by approximating a real-life problem, which may be extremely complicated, by a much simpler model.
The true f is substantially non-linear, so no matter how many training observations we are given, it will not be possible to produce an accurate estimate using linear regression. In other words, linear regression results in high bias in this example.
==Low flexible models have high bias==.
**High bias** (by fitting a horizontal line to the data).

The relative rate of change of these two quantities determines whether the test MSE increases or decreases.

The challenge lies in finding a method for which both the variance and the squared bias are low. This trade-off is one of the most important recurring themes in this book.

#### 2.2.3 The Classification Setting

The most common approach for quantifying the accuracy of our estimate ˆf is the training error rate, the proportion of mistakes that are made if we apply our estimate ˆf to the training observations:

$$\frac{1}{n} \sum_{i=1}^{n} I(y_i \neq \hat{y}_i)$$
This equation is referred to as the training error rate.

Here $\hat{y}_i$ is the predicted class label for the $i^{th}$h observation using $\hat{f}$.
And  $I(y_i \neq \hat{y}_i)$ is an indicator variable that equals 1 if $y_i \neq \hat{y}_i$ and zero if $y_i = \hat{y}_i$. indicator variable If $I(y_i \neq \hat{y}_i)$ = 0 then the $i^{th}$ observation was classified correctly by our classification method; otherwise it was misclassified.

The test error rate associated with a set of test observations of the form $(x_0, y_0)$ is given by:
$$Ave(I(y_0 \neq \hat{y}_0))$$
(2.9)
where $\hat{y}_0$ is the predicted class label that results from applying the classifier to the test observation with predictor $x_0$. A good classifier is one for which the test error (2.9) is smallest.

***The Bayes Classifier***
It is possible to show (though the proof is outside of the scope of this book) that the test error rate given in (2.9) is minimized, on average, by a very simple classifier that *assigns each observation to the most likely class, given its predictor values*.

In other words, we should simply assign a test observation with predictor vector $x_0$ to the class $j$ for which
$$Pr(Y = j|X = x_0)$$
(2.10)
is largest.
It is the probability that $Y = j$, given the observed predictor vector $x_0$. This very simple classifier is called the *Bayes classifier*.

The Bayes classifier produces the lowest possible test error rate, called the Bayes error rate.
Since the Bayes classifier will always choose the class for which (2.10) is largest, the error rate will be $1−max_j$ $Pr(Y = j|X = x_0)$ at X = $x_0$. In general, the overall Bayes error rate is given by:

$$1 − E \left( \max_j Pr(Y = j \mid X) \right)$$
where the expectation averages the probability over all possible values of X.

For our simulated data, the Bayes error rate is 0.133. It is greater than zero, because the classes overlap in the true population, which implies that $max_j$ $Pr(Y = j|X = x0) < 1$ for some values of $x_0$. The Bayes error rate is analogous to the irreducible error, discussed earlier.

***K-Nearest Neighbors***
Our goal is to make a prediction for the point labeled by the black cross. Suppose that we choose K = 3. Then KNN will first identify the three observations that are closest to the cross. This neighborhood is shown as a circle. It consists of two blue points and one orange point, resulting in estimated probabilities of 2/3 for the blue class and 1/3 for the orange class. Hence KNN will predict that the black cross belongs to the blue class.

The choice of K has a drastic effect on the KNN classifier obtained.

Using K = 1 and K = 100. When K = 1, the decision boundary is overly flexible and finds patterns in the data that don’t correspond to the Bayes decision boundary.

This corresponds to a classifier that has low bias but very high variance. As K grows, the method becomes less flexible and produces a decision boundary that is close to linear. This corresponds to a low-variance but high-bias classifier. On this simulated data set, neither K = 1 nor K = 100 give good predictions: they have test error rates of 0.1695 and 0.1925, respectively.

*In both the regression and classification settings, choosing the correct level of flexibility is critical to the success of any statistical learning method.*













