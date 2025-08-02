# A Jargon-Free ML Glossary

Friendly note: A few vocabulary terms being bolded in definitions here may only end up being defined in the related DL and LLM glossaries (to follow shortly).

1. **algorithm**: in ML terms, this usually refers to the "method" or "procedure" (like **gradient descent**) used to train a **model**, but often gets swapped in for a "**model**" itself or even a "formula" (like RMSE). You'll also hear "function" being used interchangeably. Somewhat similar to traditional CS, any differentiated meaning between of any two of these will depend on the context.

1. **anomaly detection**: an **unsupervised learning** task that looks for outliers in datasets. Commonly used for things like credit card fraud detection.

1. **asynchronous advantage actor-critic (A3C)**: A reinforcement learning **algorithm** that trains multiple agents in parallel, each exploring independently, then merges their insights into a central policy. (Think: a swarm of scouts learning faster by exploring in different directions at once.)

1. **batch learning (compare to online learning)**: a way of learning from big amounts of data, but only once in a while. Done when the **model** you are applying it to is not actively being used.

1. **Bayesian**: a way of updating what the **model** believes as new evidence comes in. The **model** starts with a guess, then adjusts that guess based on the data it sees.

1. **bias (compare with variance)**: in ML terms, means that a **model** favors a smaller subset of data than it should (underfitting). Picture:  being picky about what qualifies as a "soft drink" to mean only caffeinated beverages.

1. **bias/variance tradeoff**: ML tries to strike an optimal balance between being too simple (bias) or too sensitive (variance). Bias tends to underfit, variance tends to overfit. Probably important to add:  a **model** will always end up favoring one of these, even if slightly.

1. **boosting**: uses an **ensemble** of **models** to reduce **bias** by chaining together **model** results to improve the overall result.

1. **bucketizing**: chops a **distribution** of features into buckets of approximately equal sizes, and replacing each **feature**'s value with the bucket index ID. A good example of how ML engineers can manage things like long-tailed data.

1. **classification**: a **supervised learning** task that categorizes data into predefined classes based on labels.

1. **clustering**: an **unsupervised learning** task that results in datapoints ending up together based on their **feature** similarity to each other.

1. **correlation**: means that two or more datapoints are somehow related to each other. Doesn't mean that one datapoint causes the other though.

1. **cost function (aka loss function – compare with utility function)**: determines the distance between a **model**'s predicted values and the actual values in the **training data**. An ML Engineer's goal should be to minimize the distance.

1. **cross-validation**: assesses how well a **model** generalizes to new data.

1. **data pipeline (ML)**: an automated workflow that turns raw data into a usable **model**.

1. **decision tree**: basically if-else statements being applied to data.

1. **dimension**: typically refers to the features or attributes of the data. Perhaps easiest to think of it like a column of data in a table.

1. **dimension reduction**: reducing the number of features in a dataset while preserving what matters. The curse of dimensionality is frequently cited as a problem for learning from larger, more complex datasets.

1. **distribution**: the pattern that data gets spread out over a dataset.

2. **edge**: (especially in **graph**-based models), a connection or relationship between two **nodes** (**vertices**). Edges can be directed or undirected, and may carry weights (like similarity scores, distances, or influence strength).

1. **ensemble method**: combining multiple models together to create a better predictive or decision-based result.

1. **evaluation**: a step where a **model** is assessed on predefined metrics for performance.

1. **feature**: a measurable property of datapoints in a dataset. Think of features as "highlighted" variables which can help group datapoints in a **distribution**.

1. **feature engineering aka feature extraction**: selecting, transforming and creating new features from a raw dataset to improve the performance of a **model**.

1. **feature scaling**: used to standardize the range of features in a dataset.

1. **feature selection**: a step where the most important features will be selected to create a **model**. Sometimes handled automatically by the **algorithm** you choose.

1. **function**: in ML terms, this usually, but not always, refers to the **algorithm**.

1. **Gaussian**: another term for "normal **distribution**" aka the classic bell-shaped curve.

1. **Gaussian mixture model (GMM)**: soft **clustering** **model** that assumes data comes from a mix of several **Gaussian**-shaped blobs. Each data point belongs to each cluster a little bit, depending on how close it is. (Think:  blurry membership vs hard boundaries.)

1. **gradient**: a part sloping upward or downward (via M-W dictionary).

1. **gradient boost**: builds better predictive models by chaining together multiple weaker learners, which are typically decision trees.

1. **gradient descent**: iteratively adjusting **model** parameters to minimize a cost function.

1. **graph**: a data structure made up of **nodes** (**vertices**) and **edges** (relationships). In ML, graphs are used to represent anything with inherent connections, like social networks, molecular structures, or citation networks.

1. **graph convolutional network (GCN)**: **neural network** designed to work directly on **graphs**. Instead of pixels or sequences, it learns by passing messages along **edges** and aggregating **node** features, like learning social influence by listening to your friends.

1. **GraphSAGE**: **graph neural network** that learns how to generalize to new, unseen **nodes** by sampling and aggregating features from a node's neighbors. (Imagine learning who someone is by summarizing their social circle, even if you've never met them before.)

1. **hyperparameter (compare with parameter)**: external configuration variables that you set before **model** training begins.

1. **inference**: using a trained **model** to predict outcomes or make decisions on new, unseen data.

1. **instance**: a single complete piece of data, basically the fundamental unit that a **model** learns from.

1. **instance-based learning (compare with model-based learning)**: learns by heart—stores **training data** and compares new inputs directly (e.g., k-NN).

1. **k-nearest neighbors (k-NN)**: a **supervised learning** task used for **regression** and **classification**. Selects, for example, the 3-nearest neighbors of an **instance**. Note that "nearest" can mean different things depending on the dataset, dimensionality, and **scaling**. But it does NOT mean "sliding window".

1. **kernel**: a **function** that outputs the similarity between datapoints in a higher-dimensional space without explicit mapping. **SVM** is an example kernel method.

1. **label**: added to a piece of data to provide context for a **model**. Sometimes used interchangeably with "target", though there are nuanced differences in the literature depending on the context.

1. **linear algebra**: math for working with **vectors**, **matrices**, and linear equations. In ML, it lets us shape and manipulate data as geometric objects:  surfacing patterns, rotations, and distances that models can learn from.

1. **linear regression**: a **supervised learning** technique used to infer the relationship between a dependent variable and one or more independent variables by fitting a linear equation to the data a **model** sees.

1. **logistic regression**: a **supervised learning** technique used for **classification**, specifically for predicting the probability of a binary outcome (either 0 or 1).

1. **Jupyter notebook**: "the original web application for creating and sharing computational documents. It offers a simple, streamlined, document-centric experience" (via https://jupyter.org/)

1. **machine learning (ML)**: a branch of artificial intelligence (AI) that enables computers to predict outcomes and make decisions without ongoing human instruction, improving through experience and exposure to more data, using algorithms and statistical models to find patterns and draw inferences.

1. **Markov Chain Monte Carlo (MCMC)**: method for exploring complex probability landscapes by wandering around intelligently, collecting samples that reflect the true **distribution**. Used when you can't solve a **Bayesian** posterior with math, so you sample your way to understanding it.

1. **matrix**: a two-dimensional array of numbers (aka scalars) organized into rows and columns.

1. **mean absolute error (MAE)**: quantifies the average magnitude of errors between predicted and actual values in a dataset. Commonly used in **regression** tasks to measure **model** accuracy in units that match the target variable.

1. **min-max scaling**: rescales values to a fixed range, usually [0, 1], based on the min and max of the **feature**.

1. **model**: a program that learns from some data to make predictions or decisions about similar data. Sometimes also referred to as a function or **algorithm**.

1. **model-based learning (compare to instance-based learning)**: learns by pattern. Builds a general **model** from the **training data** to make predictions.

1. **multilayer perceptron (MLP)**: classic feedforward neural network:  layers of neurons, weighted sums, activation functions. No tricks, no loops, just pure dense function approximation. Works best on tabular data and small inputs.

1. **multivariate regression vs. univariate regression**: univariate predicts a single output from a single input. Multivariate predicts one or more outputs using multiple inputs.

1. **naïve Bayesian**: a **classification** method that assumes every **feature** is independent ("naïve") and uses a **Bayesian** approach to learning.

1. **normalization**: rescales data to fit within a fixed range, usually [0, 1]. Useful when features have different units or scales (like measuring people's height in inches and weight in pounds), so that no single **feature** dominates the **model**.

1. **novelty detection**: spotting datapoints that look different from anything the **model** has seen before. Helpful for catching things like fraud, defects, or weird behavior.

1. **online learning (compare with batch learning)**: where the **model** updates itself incrementally as new datapoints arrive, either one at a time or in small batches. Unlike batch learning, it doesn't retrain on the whole dataset. It is not actually done "online" in the normal meaning of "online" though.

1. **out-of-core learning**: when the dataset is too large for a single machine's memory, models must learn from the data in chunks. A classic example of real-world MLOps complexity.

1. **overfitting (data – compare with underfitting)**: when a **model** shows low **bias** but **high variance**, it fits the **training data** too closely, including noise or outliers. As a result, it struggles to generalize to new, unseen data. Means your **model** is catching more datapoints than it should (like including "bobcats" in "house cats" because "cats").

1. **parameter (compare with hyperparameter)**: internal variables within a **model** that are adjusted during training to improve **model** performance. (Technically, parameters are learned from the data.)

1. **perceptron**: the simplest type of **neural network** unit, and one of the earliest. Takes multiple inputs, applies **weights**, adds a **bias**, and runs the result through an **activation function** (typically a step or threshold function).

1. **polynomial regression**: fits curves instead of straight lines by adding powers of input variables (e.g., x^2, x^3). Useful when relationships aren't linear.

1. **predictor**: a variable or **model** used to guess or estimate the value of something else. In ML terms, it's the thing doing the predicting, like a **model** output or an input **feature** that helps forecast a target.

1. **principal component analysis (PCA)**: a dimensionality reduction tool that rotates and flattens data to find the axes with the most **variance**. (Think:  squashing 3D shadows onto a 2D wall in the most informative way possible.)

1. **proximal policy optimization (PPO)**: a reinforcement learning **algorithm** that improves policy updates by keeping changes small and safe.

1. **random forest**: an **ensemble method** that builds a bunch of decision trees and then averages their results. Each tree "votes", and the forest makes the final call, usually providing better accuracy, less overfitting.

1. **regression**: in ML terms, a **supervised learning** technique used to predict a continuous numerical value based on the **model**'s input features. (Comes from statistics, where "regression to the mean" describes how extreme data observations tend to move closer to an average across future observations. ML regression inherits this framing, though in practice it's about fitting functions that minimize error between predicted and actual values.)

1. **regularization**: adds a penalty to the **model**'s complexity so it doesn't overfit weird quirks or noise in the data. It doesn't actually delete data, but it does discourage the **model** from obsessing over outliers. (To actually remove outliers, you'd use a different technique like robust **regression** or data filtering.)

1. **reinforcement learning (RL)**: an agent learns by trial and error, getting rewards or penalties based on the actions it takes in an environment.

1. **root mean squared error (RMSE)**: measures how far predictions are from actual values. Larger errors get punished more than smaller ones (because it squares first). Tries to show how much error a **model** makes in a prediction.

1. **scalar (compare with vector)**: in ML terms, a piece of data that has a single numeric value, e.g., "2.0".

1. **scaling**: a preprocessing step that adjusts the range of values in your data, so no single **feature** dominates. Often means shrinking every value to a common scale, like between 0 and 1.

1. **Scikit-Learn**: a free open-source machine learning library for Python. Provides a wide range of tools and algorithms for core ML task types like **classification**, **regression**, **clustering**, and dimensionality reduction.

1. **self-supervised learning**: the **model** learns from raw data without needing human-labeled examples, by generating its own **labels** from the data itself. (Example:  predicting the next word in a sentence. The words are the input and the target.)

1. **semi-supervised learning**: applies a small amount of **labeled** data to a large amount of unlabeled data. Tries to stretch the value of human-labeled examples across the rest of the dataset.

1. **SimCLR**: a **self-**supervised learning**** method for images. It learns by comparing differently augmented versions of the same image, but without needing labels. (Core trick:  bring similar views closer in embedding space, push others apart.)

1. **softmax**: turns groups of values in a dataset into a list of probabilities that sum to 1 (e.g., [0.68, 0.27, 0.05]).

1. **standardization**: rescales data to have a mean of 0 and a standard deviation of 1. Unlike **normalization**, which squishes values to a specific range, **standardization** centers and balances them for models that assume bell-shaped distributions (like **linear **regression**** or **logistic **regression****).

1. **supervised learning**: the **model** learns from labeled examples. Each input in the **training data** comes with the correct answer, and the **model** tries to learn the mapping from input to output.

1. **support vector machine (SVM)**: method that tries to find the best boundary (or "hyperplane", in ML jargon) between different groups of data. It focuses on the hardest-to-classify points, the ones right near the edge.

1. **support vector regression (SVR)**: predicts continuous values by fitting a function (or "hyperplane", in ML jargon) to a dataset, aiming to minimize the error while keeping as many datapoints within a specified margin.

1. **test data**: the slice of your data you DON'T train on. Used only to check how well your trained **model** may perform on new, unseen examples. Typically 20% of smaller datasets, less for larger datasets.

1. **training (ML)**: the process of adjusting a **model**'s internal parameters based on **training data**, usually by minimizing a loss function, so it gets better at making predictions. In **supervised learning**, this means learning from labeled examples.

1. **training data**: the part of your dataset the **model** DOES see during training, it's what the **model** uses to learn. Typically 80% of smaller datasets, more for larger datasets.

1. **transfer learning**: at a basic level, all ML is transfer, because you're transferring what you've learned from **training data** to any new data your **model** sees later. (Lately the term "**transfer learning**" is bearing some extra load as a DL term, with a more nuanced meaning, which we will explore in the DL Foundation course ahead.)

1. **underfitting (data – compare with overfitting)**: when a **model** shows high bias and low variance, it's too simple to capture the real patterns in the data, so it performs poorly on both training and test sets. Means your **model** isn't catching as many datapoints as it should (like excluding "gray tigers" from "house cats" because "tigers").

1. **unsupervised learning**: works on unlabeled data. The **model** tries to find structure or patterns (like clusters or groupings), as well as outliers, without being told what the "right" answer is.

1. **utility function (aka fitness function – compare with cost function)**: a score that tells the **model** how well it's doing. Higher = better. A "reward" signal that helps guide the **model** toward better decisions by quantifying success.

1. **variance (compare with bias)**: in ML terms, means that a **model** favors a larger subset of data than it should (overfitting). Picture:  spreading out a wide variance which allows "soft drink" to mean anything else near the carbonated beverages in a supermarket refrigerator.

1. **vector (compare with scalar)**: in ML terms, a piece of data that has multiple numeric values e.g., [2.0, 5].

1. **vertex (plural**: vertices): in ML terms, another word for a **node** in a **graph**. It's a single entity in the network, like a person in a social graph, or a protein in a biological interaction map.

1. **vision transformer (ViT)**: a transformer architecture applied to images instead of text. It chops images into patches, treats them like tokens, and uses self-attention to learn how different parts relate. Surprisingly effective despite ignoring local pixel structure.

1. **weight**: in ML terms, a tunable **parameter** that represents the importance of a specific input to a **model's** prediction. During **training**, weights are adjusted so the model gets better at minimizing its error.

1. **z-score**: represents how many standard deviations a data point is away from the mean of a dataset.
