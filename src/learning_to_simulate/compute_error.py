import math
import numpy as np

def normalizeRMSE(err, min_d, max_d):
    rmse = math.sqrt(err)
    print("rmse", rmse)
    normalized_err = (rmse - min_d) / (max_d - min_d)
    print("normalized err", normalized_err)
    force_err = 48.63 * normalized_err # one unit of force in simulation is 48.63 pN
    return force_err

def standardizeRMSE(err, std, mean):
    rmse = math.sqrt(err)
    standardized_err = (rmse * std) + mean
    force_err = 48.63 * standardized_err
    return force_err

if __name__ == "__main__":
    # cuboid c3
    # err = 16.977 # one step position mse
    # err = 0.01564 # loss mse
    # cuboid c1
    # err = 0.4986 # one step position mse
    # err = 0.0004768 # loss mse
    # cuboid c05
    # err = 0.3152 # one step position mse
    # err = 0.0003074 # loss mse

    # # cuboid
    # cmean = np.mean([[-0.00059495, -0.00706764, 0.01329268]])
    # cstd = np.std([[33.06003383, 33.01174007, 32.67817926]])
    
    # llama
    cmean = np.mean([[-0.01117366, 0.01696055, -0.01233886]])
    cstd = np.std([[32.33672049, 32.66093383, 32.72852048]])

    # # scotty
    # cmean = np.mean([[0.02886615, 0.0077857, 0.0263394]])
    # cstd = np.std([[31.56985564, 32.4151203, 31.43540538]])

    # scotty c05
    # err = 0.2722 # one step position mse
    # err = 0.0002835 # loss mse
    # scotty c1
    # err = 0.7061 # one step position mse
    # err = 0.0007022 # loss mse
    # scotty c3
    # err = 17.934 # one step position mse
    # err = 0.01803 # loss mse

    # llama c05
    # err = 0.1044 # one step position mse
    # err = 0.0001078 # loss mse
    # llama c1
    # err = 0.2219 # one step position mse
    # err = 0.0002161 # loss mse    
    # # llama c3
    # err = 15.596 # one step position mse
    err = 0.01471 # loss mse

    # # cuboid
    # min_d = -464.14
    # max_d = 414.46

    # llama
    # min_d = -414.97
    # max_d = 397.93

    # # scotty
    # min_d = -389.72
    # max_d = 347.39
    
    print("The err is ", err)
    # force_err = normalizeRMSE(err, min_d, max_d)
    force_err = standardizeRMSE(err, cstd, cmean)
    print("The force error is ", force_err)