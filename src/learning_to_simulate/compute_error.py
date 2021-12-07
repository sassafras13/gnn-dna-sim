import math

def normalizeRMSE(err, min_d, max_d):
    rmse = math.sqrt(err)
    print("rmse", rmse)
    normalized_err = (rmse - min_d) / (max_d - min_d)
    print("normalized err", normalized_err)
    force_err = 48.63 * normalized_err # one unit of force in simulation is 48.63 pN
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
    min_d = -414.97
    max_d = 397.93

    # # scotty
    # min_d = -389.72
    # max_d = 347.39
    
    print("The err is ", err)
    force_err = normalizeRMSE(err, min_d, max_d)
    print("The force error is ", force_err)