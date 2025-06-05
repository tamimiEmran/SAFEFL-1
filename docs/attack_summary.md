# Federated Learning Attack Methods Summary

This document provides a detailed explanation of the attack methods implemented in `attacks.py`.

## No Attack (`no_byz`)

- **Description**: This is a baseline function that performs no attack, simply returning the original gradients unchanged.
- **Operation**: Returns the input gradients without modification.
- **Use Case**: Used as a control/baseline for evaluating aggregation rule performance.

## Trim Attack (`trim_attack`)

- **Description**: An attack specifically designed to compromise the trimmed mean aggregation rule.
- **Operation**: 
  1. Calculates the general direction of the aggregate gradients by finding the sign of the sum of all gradients.
  2. Identifies maximum and minimum values across all dimensions.
  3. Crafts malicious gradients by manipulating values using the component-wise direction and extremes, applying random scaling to exacerbate the effect.
  4. Directs malicious gradients to either maximize or minimize the trimmed mean (depending on the direction).
- **Target**: Trimmed mean aggregation rule.
- **Reference**: [Little Is Enough: Byzantine-Resilient Federated Learning via Feature Distillation](https://arxiv.org/abs/1911.11815)

## Krum Attack (`krum_attack`)

- **Description**: A model poisoning attack targeting the Krum aggregation rule.
- **Operation**:
  1. Calculates Euclidean distances between all client updates.
  2. Computes the minimum sum of distances among benign clients.
  3. Determines the maximum scaled distance between benign clients and the model.
  4. Uses these values to create an optimal negative direction for malicious updates.
  5. Gradually refines a scaling factor (lambda) until malicious updates are selected by Krum.
  6. Creates adversarial gradients in the opposite direction of honest gradients.
- **Target**: Krum aggregation rule.
- **Reference**: [Little Is Enough: Byzantine-Resilient Federated Learning via Feature Distillation](https://arxiv.org/abs/1911.11815)

## FLTrust Attack (`fltrust_attack`)

- **Description**: A sophisticated attack against the FLTrust aggregation rule (also known as adaptive attack).
- **Operation**:
  1. Normalizes all gradients and calculates the cosine similarity between each client's gradient and the server's gradient.
  2. Computes the weighted sum of client gradients based on these similarities.
  3. Uses a combination of optimization techniques to generate adversarial updates.
  4. Employs a stochastic gradient descent procedure to find malicious gradients that maximize the negative impact on model performance.
  5. Scales the final adversarial gradients by the norm of the server's gradient.
- **Target**: FLTrust aggregation rule.
- **Reference**: [FLTrust: Byzantine-robust Federated Learning via Trust Bootstrapping](https://arxiv.org/abs/2012.13995)

## Min-Max Attack (`min_max_attack`)

- **Description**: An optimization-based attack that maximizes the distance between the aggregated result and the true gradient mean.
- **Operation**:
  1. Calculates the mean of all gradient updates.
  2. Computes a deviation direction (normalized mean gradient).
  3. Uses a binary search algorithm to find the optimal scaling factor (gamma).
  4. Crafts malicious updates by moving in the opposite direction of the mean with the optimal strength.
  5. Ensures the crafted updates remain within a distance constraint to avoid detection.
  6. Optimizes to maximize the distance between malicious updates and benign ones.
- **Target**: General aggregation rules, particularly effective against geometric median-based methods.
- **Reference**: [Fall of Empires: Breaking Byzantine-tolerant SGD by Inner Product Manipulation](https://par.nsf.gov/servlets/purl/10286354)

## Min-Sum Attack (`min_sum_attack`)

- **Description**: Similar to Min-Max but optimizes for minimizing the sum of distances instead of maximizing the maximum distance.
- **Operation**:
  1. Similar to Min-Max attack, but uses a different optimization criterion.
  2. Calculates the mean of all gradient updates and a deviation direction.
  3. Performs binary search to find the optimal scaling factor.
  4. Creates malicious updates that minimize the sum of distances between updates.
  5. More effective against aggregation rules that use the sum of distances rather than maximum distance.
- **Target**: Distance-based aggregation rules, especially those using sum of distances in their calculations.
- **Reference**: [Fall of Empires: Breaking Byzantine-tolerant SGD by Inner Product Manipulation](https://par.nsf.gov/servlets/purl/10286354)

## Label Flipping Attack (`label_flipping_attack`)

- **Description**: A data poisoning attack that manipulates training data labels on malicious clients.
- **Operation**:
  1. For each malicious client, inverts the labels of their training data.
  2. Maps each label `l` to `num_labels - l - 1` (e.g., in a 10-class problem, 0→9, 1→8, etc.).
  3. Forces the model to learn incorrect associations between features and labels.
- **Target**: The learning process itself, not a specific aggregation rule.
- **Effect**: Decreases overall model accuracy and reliability.

## Scaling Attack

This is a two-part attack combining data poisoning with gradient scaling:

### Part 1: Backdoor Insertion (`scaling_attack_insert_backdoor`)

- **Description**: Inserts backdoor patterns into training data on malicious clients.
- **Operation**:
  1. Selects a portion of training data on malicious clients.
  2. Adds a trigger pattern to these samples (e.g., setting specific pixels/features to fixed values).
  3. Changes labels of trigger-containing samples to an attacker-chosen target label.
  4. For MNIST, adds a white square pattern in the top-left corner as the trigger.
  5. For HAR, sets specific features to zero as the trigger.

### Part 2: Gradient Scaling (`scaling_attack_scale`)

- **Description**: Amplifies the impact of malicious gradients to influence the aggregation result.
- **Operation**:
  1. Multiplies malicious clients' gradients by a scaling factor (equal to the total number of clients).
  2. Makes malicious gradients contribute disproportionately to the aggregated result.

- **Target**: Primarily effective against simple averaging (FedAvg), but can affect other rules.
- **Effect**: Creates a backdoor in the model that misclassifies inputs with the trigger pattern.
- **Reference**: [FLTrust: Byzantine-robust Federated Learning via Trust Bootstrapping](https://arxiv.org/abs/2012.13995)

## Backdoor Testing Function (`add_backdoor`)

- **Description**: A utility function that adds trigger patterns to test data to evaluate backdoor attack success.
- **Operation**:
  1. Adds the same trigger pattern used in the scaling attack to test inputs.
  2. Sets all labels to the attacker's chosen target label.
  3. Used during evaluation to measure the backdoor success rate.