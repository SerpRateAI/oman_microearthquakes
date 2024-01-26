
## Perform adaptive downsampling of the velocity profile

def adaptive_downsample(depax, vels_in, threshold, ddep_min, ddep_max):
    from numpy import array, abs, min, max

    numdep = len(depax)
    ddep_in = depax[1] - depax[0]
    ind = 0
    deps_out = []
    vels_out = []

    while ind < numdep-1:
        dep = depax[ind]
        vel = vels_in[ind]
        deri = get_forward_derivative(depax, vels_in, ind)

        deps_out.append(dep)
        vels_out.append(vel)

        # Determine the next step based on the derivative
        ddep = max([ddep_min, min([ddep_max, threshold / abs(deri + 1e-10)])])
        step = int(ddep / ddep_in)
        ind += step

    return deps_out, vels_out

## Compute the forward derivative
def get_forward_derivative(x, y, ind):
    derivative = (y[ind + 1] - y[ind]) / (x[ind + 1] - x[ind])
    return derivative

## Convert vp to density the empirical relation in Brocher (2005)
def vp2rho(pvels):
    p1 = 1.6612;
    p2 = -0.4721;
    p3 = 0.0671;
    p4 = -0.0043;
    p5 = 0.000106;

    rhos = p1*pvels+p2*pvels**2+p3*pvels**3+p4*pvels**4+p5*pvels**5

    return rhos
