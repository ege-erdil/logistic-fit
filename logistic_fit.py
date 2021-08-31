from autograd import grad
import autograd.numpy as np
from scipy.stats import logistic, norm
from scipy.optimize import minimize

def logistic_pdf(x, loc, scale):
    y = (x - loc)/scale
    return np.exp(-y)/(scale * (1 + np.exp(-y))**2)

def logistic_cdf(x, loc, scale):
    y = (x-loc)/scale
    if y < -100:
        return 0
    elif y > 100:
        return 1
    else:
        return 1/(1 + np.exp(-y))

def logistic_logpdf(x, loc, scale):
    y = (x - loc)/scale
    if y < -250:
        return y - np.log(scale)
    elif y > 250:
        return -y - np.log(scale)
    else:
        return -y - np.log(scale) - 2 * np.log(1 + np.exp(-y))

def square_dist(a1, a2):
    s = 0
    for k in range(len(a1)):
        s += (a1[k] - a2[k])**2
    return s

def log_likelihood_logistic(data, params):
    n = len(data)
    c = (len(params) + 1)//3
    r = 0

    if (len(params) + 1) % 3 != 0:
        print("Parameters specified incorrectly!")
        return None

    else:
        weights = [1]
        for k in range(c-1):
            weights.append(np.exp(params[2*c + k]))
        s = np.sum(weights)
        for x in data:
            pdf_list = [logistic_logpdf(x, params[2*j], np.exp(params[2*j+1])) for j in range(c)]
            pdf_list_avg = np.sum(pdf_list)/c
            pdf_list_n = [weights[j] * np.exp(pdf_list[j] - pdf_list_avg) for j in range(c)]
            
            r += (pdf_list_avg + np.log(np.sum(pdf_list_n)/s))/n
        return r

def cdf_loss(percentiles, params):
    n = len(percentiles)
    c = (len(params) + 1)//3
    r = 0

    if (len(params) + 1) % 3 != 0:
        print("Parameters specified incorrectly!")
        return None

    else:
        weights = [1]
        for k in range(c-1):
            weights.append(np.exp(params[2*c + k]))
        s = np.sum(weights)
        for q in range(1, n):
            cdf_list = [logistic_cdf(percentiles[q-1], params[2*j], np.exp(params[2*j+1])) for j in range(c)]
            cdf_list_n = [weights[j] * cdf_list[j] for j in range(c)]
            
            r += (np.sum(cdf_list_n)/s - q/n)**2/n
        return r


def estimate(data, bins=20, num = 1, tol = 0.01, maxiter = 100):
    fit_params = np.zeros(3*num - 1)
    a = np.average(data)
    s = np.log(np.std(data))
    percentiles = [np.percentile(data, k) for k in range(100//bins, 100, 100//bins)]
    for i in range(num):
        fit_params[2*i] = np.random.normal(loc=a, scale=np.exp(s), size=1)
        fit_params[2*i+1] = np.random.normal(loc=s - np.log(num), scale=1, size=1)

    def training_loss(params):
        return cdf_loss(percentiles, params) + 0.0001 * np.dot(params[2*num:], params[2*num:])
    
    training_loss_jac = grad(training_loss)

    res = minimize(training_loss, jac=training_loss_jac, x0=fit_params, method="BFGS", options = {"maxiter": maxiter, "gtol": tol})
    print(res)
    final_params = res.x
    for i in range(num):
        final_params[2*i+1] = np.exp(final_params[2*i+1])
    results = []
    for i in range(num):
        results.append(final_params[2*i])
        results.append(logistic.isf(0.25, loc=final_params[2*i], scale=final_params[2*i+1]) - final_params[2*i])

    for i in range(num-1):
        results.append(final_params[2*num + i])

    return results

def estimate_log(data, num = 1, tol = 0.01, maxiter = 100):
    fit_params = np.zeros(3*num - 1)
    a = np.average(data)
    s = np.log(np.std(data))
    for i in range(num):
        fit_params[2*i] = np.random.normal(loc=a, scale=np.exp(s), size=1)
        fit_params[2*i+1] = np.random.normal(loc=s - np.log(num), scale=1, size=1)
        
    def training_likelihood(params):
        return log_likelihood_logistic(data, params)

    def training_loss(params):
        return -log_likelihood_logistic(data, params)
    
    training_likelihood_jac = grad(training_likelihood)
    training_loss_jac = grad(training_loss)

    res = minimize(training_loss, jac=training_loss_jac, x0=fit_params, method="BFGS", options = {"maxiter": maxiter, "gtol": tol})
    print(res)
    final_params = res.x
    for i in range(num):
        final_params[2*i+1] = np.exp(final_params[2*i+1])
    results = []
    for i in range(num):
        results.append(final_params[2*i])
        results.append(logistic.isf(0.25, loc=final_params[2*i], scale=final_params[2*i+1]) - final_params[2*i])

    for i in range(num-1):
        results.append(final_params[2*num + i])

    return results

def estimate_powell(data, num = 1, tol = 0.01, maxiter = 100):
    fit_params = np.zeros(3*num - 1)
    a = np.average(data)
    s = np.log(np.std(data))
    for i in range(num):
        fit_params[2*i] = np.random.normal(loc=a, scale=np.exp(s), size=1)
        fit_params[2*i+1] = np.random.normal(loc=s - np.log(num), scale=1, size=1)
        
    def training_likelihood(params):
        return log_likelihood_logistic(data, params)

    def training_loss(params):
        return -log_likelihood_logistic(data, params)
    
    training_likelihood_jac = grad(training_likelihood)
    training_loss_jac = grad(training_loss)

    res = minimize(training_loss, x0=fit_params, method="Powell", tol=tol, options = {"maxiter": maxiter})
    print(res)
    final_params = res.x
    for i in range(num):
        final_params[2*i+1] = np.exp(final_params[2*i+1])
    results = []
    for i in range(num):
        results.append(final_params[2*i])
        results.append(logistic.isf(0.25, loc=final_params[2*i], scale=final_params[2*i+1]) - final_params[2*i])

    for i in range(num-1):
        results.append(final_params[2*num + i])

    return results
