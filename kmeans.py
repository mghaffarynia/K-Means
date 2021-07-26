
import pandas as pd
import numpy as np
import math, statistics, functools
import matplotlib.pyplot as plt
from sklearn.metrics.cluster import adjusted_rand_score

def read_csv_input(filename):
  df = pd.read_csv(filename, header = None).to_numpy()
  y = df[:, [0]]
  X = df[:, range(1, df.shape[1])]
  return X, y

def normalize(X):
  means = np.mean(X, axis=0)
  stds = np.std(X, axis=0)
  return (X-means)/stds, means, stds

def kmeans_random_init(n, k, low=-3, high=3):
  return np.random.uniform(low, high, k*n).reshape(k, n)

def gmm_random_init(m, n, k):
  means = np.random.multivariate_normal(np.zeros(n), np.identity(n), k)
  return means

def kmeans_plus_plus_init(X, k):
  m, n = X.shape
  means = np.zeros((k, n))
  means[0, :] = X[np.random.choice(m), :]
  for i in range(1, k):
    dis = distance(X, means[:i, :])
    nearest_dis_sq = np.power(np.min(dis, axis=1), 2)
    pi = nearest_dis_sq / np.sum(nearest_dis_sq)
    lucky = np.random.choice(np.arange(m), 1, p=pi)
    means[i, :] = X[lucky, :]
  return means

def calc_probs(X, means, cov_matrix):
  m, n = X.shape
  X_diff = X[None, :, :] - means[:, None, :]
  var = np.sum(np.matmul(X_diff, np.linalg.inv(cov_matrix)) * X_diff, axis = -1)
  numer = np.exp(-0.5 * var).T
  denom = ((2*np.pi)**(n/2))*(np.linalg.det(cov_matrix)**(0.5))
  return numer/denom

def e_step(X, means, cov_matrix, lams):
  probs = calc_probs(X, means, cov_matrix)
  lam_probs = probs * lams
  q = lam_probs / np.sum(lam_probs, axis = 1, keepdims=True)
  return q

def m_step(X, q):
  m, n = X.shape
  _, k = q.shape
  q_3d = np.broadcast_to(q, (1, m, k)).T
  X_3d = np.broadcast_to(X, (k, m, n))
  numer = np.sum(q_3d * X_3d, axis=1)
  denom = np.sum(q_3d, axis=1)
  means = numer/denom
  X_diff = X_3d - means[:, None, :]
  cov_matrix = np.matmul(np.transpose(q_3d * X_diff, (0, 2, 1)), X_diff)/denom[:, :, None]
  cov_matrix += (np.identity(n)*1e-6)[None, :, :]
  lams = np.sum(q, axis=0)/m
  return means, cov_matrix, lams

def gmm(X, k, init_means):
  m, n = X.shape
  means = init_means
  cov_matrix = np.broadcast_to(np.identity(n), (k, n, n))
  lams = np.ones(k)/k
  losses = []
  while True:
    q = e_step(X, means, cov_matrix, lams)
    means, cov_matrix, lams = m_step(X, q)
    probs = calc_probs(X, means, cov_matrix)
    loss = np.sum(np.log(np.sum(probs * lams, axis=1)))
    losses.append(loss)
    if len(losses) > 2 and abs(loss - losses[-2]) < 1e-3:
      break
  lam_probs = probs * lams
  classes = np.argmax(lam_probs, axis=1)
  return means, cov_matrix, lams, losses, classes

def distance(X1, X2):
  X1_sq = np.diag(np.dot(X1, X1.T)).reshape(X1.shape[0], -1)
  X2_sq = np.diag(np.dot(X2, X2.T)).reshape(-1, X2.shape[0])
  X1X2_2 = 2*np.dot(X1, X2.T)
  dist = X1_sq + X2_sq - X1X2_2
  return dist

def kmeans(X, k, init_centroids):
  m, n = X.shape
  prev_clusters = np.zeros(m) - 1
  centroids = init_centroids
  objectives = []
  while True:
    dis = distance(X, centroids)
    objectives.append(np.sum(np.min(dis, axis = 1))/m)
    clusters = np.argmin(dis, axis = 1)
    for i in range(k):
      if np.sum(clusters == i) > 0:
        centroids[i, :] = np.mean(X[clusters == i, :], axis = 0)
    if np.all(clusters == prev_clusters):
      break
    prev_clusters = clusters
  return clusters, centroids, objectives

def run_kmeans(X, ks, init_fun):
  print("######################")
  print("K-Means:")
  _, n = X.shape
  for k in ks:
    print("~~~~~~~~~~~~~~~~~~~~~")
    print(f"For k: {k}")
    loss_vals = []
    for r in range(20):
      init_centroids = init_fun(k)
      clusters, centroids, losses = kmeans(X, k, init_centroids)
      loss_vals.append(losses[-1])
    mean, var = statistics.mean(loss_vals), statistics.variance(loss_vals)
    print(f"\tmean:     {mean}\n\tvariance: {var}\n")

def run_gmms(X, ks, init_fun):
  print("######################")
  print("Gaussian Mixture Model:")
  for k in ks:
    print("~~~~~~~~~~~~~~~~~~~~~")
    print(f"For k: {k}")
    loss_vals = []
    for r in range(20):
      init_means = init_fun(k)
      means, cov_matrix, lams, losses, classes = gmm(X, k, init_means)
      loss_vals.append(losses[-1])
    mean, var = statistics.mean(loss_vals), statistics.variance(loss_vals)
    print(f"\tmean:     {mean}\n\tvariance: {var}\n")

def compare_clusterings(X, k, kmeans_init_fun, gmm_init_fun, R=20):
  print("######################")
  print(f"Comparing Kmeans and GMM for k={k}:")
  aris = [[], []]
  for r in range(R):
    init_centroids = kmeans_init_fun(k)
    kmeans_clusters, _, _ = kmeans(X, k, init_centroids)
    init_means = gmm_init_fun(k)
    _, _, _, _, gmm_clusters = gmm(X, k, init_means)
    aris[0].append(adjusted_rand_score(y[:, 0].tolist(), kmeans_clusters))
    aris[1].append(adjusted_rand_score(y[:, 0].tolist(), gmm_clusters))
  kmeans_ari_mean = statistics.mean(aris[0])
  gmm_ari_mean = statistics.mean(aris[1])
  print(f"\tK-Means Similarity Rand Score: {kmeans_ari_mean}")
  print(f"\tGMM Similarity Rand Score:     {gmm_ari_mean}\n")

def main():
  X, y = read_csv_input('leaf.data')
  m, n = X.shape
  X_norm, mu, sigma = normalize(X)
  ks = [12, 18, 24, 36, 42]

  kmean_rand_fun = functools.partial(kmeans_random_init, n)
  gmm_rand_fun = functools.partial(gmm_random_init, m, n)
  kmeans_pp_fun = functools.partial(kmeans_plus_plus_init, X_norm)

  print("WITH RANDOM INITIALIZATION\n")
  run_kmeans(X_norm, ks, kmean_rand_fun)
  run_gmms(X_norm, ks, gmm_rand_fun)
  compare_clusterings(X_norm, 36, kmean_rand_fun, gmm_rand_fun)

  print("\n\nWITH KMEANS++ INITIALIZATION\n")
  run_kmeans(X_norm, ks, kmeans_pp_fun)
  run_gmms(X_norm, ks, kmeans_pp_fun)
  compare_clusterings(X_norm, 36, kmeans_pp_fun, kmeans_pp_fun)

main()