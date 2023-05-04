import os
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from   matplotlib import cm
import george
from george import kernels
from scipy.optimize import minimize
from scipy.stats import multivariate_normal
from functools import partial
import statistics
from   mpl_toolkits.mplot3d import Axes3D
import time as ti
import corner
import time

# class astronomical_object():

bands = ['g', 'r', 'tess']

def get_data(filename):
        """
        extract data from the file
        """
        f = pd.read_csv(filename)

        tess = f[['relative_time', 'tess_flux', 'tess_uncert']]
        r = f[['relative_time', 'r_flux', 'r_uncert']]
        g = f[['relative_time', 'g_flux', 'g_uncert']]

        tess = tess.rename(columns={"tess_flux": "flux", "tess_uncert": "uncert"})
        r = r.rename(columns={"r_flux": "flux", "r_uncert": "uncert"})
        g = g.rename(columns={"g_flux": "flux", "g_uncert": "uncert"})

        tess['band'] = 'tess'
        r['band'] = 'r'
        g['band'] = 'g'

        observations = pd.concat([tess, r, g], ignore_index=True)
        observations = observations.dropna()
        observations = observations.sort_values(by=['relative_time'])

        # ???
        observations = observations.loc[(observations['relative_time'] >= -30) & (observations['relative_time'] <= 70)]

        return observations, f[['mwebv', 'max_light', 'max_uncert']]

def fit_gaussian_process(gp_observations, guess_length_scale = 10, prior = False):
        """
        Fit a Gaussian Process model to the light curve.
        We use a 2-dimensional Matern kernel to model the transient. The kernel
        width in the wavelength direction is fixed. We fit for the kernel width
        in the time direction as different transients evolve on very different
        time scales.
        Parameters
        ----------
        gp_observations:pandas.Dataframe

        fix_scale : bool (optional)
            If True, the scale is fixed to an initial estimate. If False
            (default), the scale is a free fit parameter.
        guess_length_scale : float (optional)
            The initial length scale to use for the fit. The default is 10
            days.
        prior:

        Returns
        -------
        gaussian_process : function
            A Gaussian process conditioned on the object's lightcurve. This is
            a wrapper around the george `predict` method with the object flux
            fixed.
        partial_sample : pandas.DataFrame
            The processed observations that the GP was fit to. This could have
            effects such as background subtraction applied to it.
        result.x : list
            A list of the resulting GP fit parameters.
        """
        flux = gp_observations['flux']
        flux_err = gp_observations['uncert']
        
        wavelengths = gp_observations['band'].map(get_band_central_wavelength)
        time = gp_observations['relative_time']

        # Use the highest signal-to-noise observation to estimate the scale. We
        # include an error floor so that in the case of very high
        # signal-to-noise observations we pick the maximum flux value.
        signal_to_noises = np.abs(flux) / np.sqrt(
            flux_err ** 2 + (1e-2 * np.max(flux)) ** 2
        )
        scale = np.abs(flux[signal_to_noises.idxmax()])

        kernel = (0.5 * scale) ** 2 * kernels.Matern32Kernel(
            [guess_length_scale ** 2, 6000 ** 2], ndim=2
        )


        # freeze wavelength parameter
        kernel.freeze_parameter("k2:metric:log_M_1_1")

        gp = george.GP(kernel)
        # print(gp.get_parameter_vector())
        # print(gp.get_parameter_names())
        guess_parameters = gp.get_parameter_vector()
        x_data = np.vstack([time, wavelengths]).T
        gp.compute(x_data, flux_err)

        def neg_ln_like(p):
            gp.set_parameter_vector(p)
            return -gp.log_likelihood(flux)

        def grad_neg_ln_like(p):
            gp.set_parameter_vector(p)
            return -gp.grad_log_likelihood(flux)

        def log_prior(p):
            """
            p should be a vector of your two parameters for constant and time (same input as for log_likelihood)
            """
            n = 2

            # The means for the constant and time parameter
            means = np.array([10.75613775, 5.78162741])

            # The covariance matrix from np.cov you calculated earlier
            covariance = np.array([[2.56262992 ,0.03282953], 
                                    [0.03282953, 0.73455561]])

            # You can compute the inverse of the matrix using https://numpy.org/doc/stable/reference/generated/numpy.linalg.inv.html and then store the result here just like you stored the result of the means and covariance above
            inverse_covariance = np.array([[ 0.39044767, -0.0174503 ],
                                            [-0.0174503,   1.36214723]])
            

            # The @ symbol does matrix multiplication
            logprior = -(n/2)*np.log(2*np.pi) - 0.5*np.log(np.linalg.det(covariance)) - 0.5 * (p-means) @ inverse_covariance @ (p-means)

            # The 1D version of the above equation would use "sigmas" (which would just be an array with two numbers) instead of covariance. And the "@" would be replaced by multiplications and would simply reflect the equation we looked at

            # if p[1] < 4:
            #     return np.inf
            return logprior

        def neg_log_posterior(p):
            return neg_ln_like(p) - log_prior(p)

        time_bounds = [(3.6, 8)] #[(1, np.log(1000 ** 2))]               # bounds for time scale
        wavelength_bounds = [(None, None)]
        bounds = [(3, 18)] + time_bounds #[(guess_parameters[0] -10, 16)] + time_bounds #+ wavelength_bounds          # bounds for constant and time scale

        # if prior:
        result = minimize(neg_log_posterior, [10.75,5.45], bounds=bounds, method='Nelder-Mead', options={'xatol': 1e-12})
        # else:
        # result = minimize(neg_ln_like, gp.get_parameter_vector(), jac=grad_neg_ln_like, bounds=bounds)

        gaussian_process = partial(gp.predict, flux)
        
        partial_sample = partial(gp.sample_conditional, flux)

        return gaussian_process, gp_observations, result.x

def predict_gaussian_process(gp_observations, bands, t_pred, fitted_gp=None, fitted_sample = None, prior = False):  
    if fitted_gp is not None:
        # gp, sample = fitted_gp, fitted_sample
        gp = fitted_gp
    else:
        # gp, sample, _ = fit_gaussian_process(gp_observations, prior = prior)
        gp, _, _ = fit_gaussian_process(gp_observations, prior = prior)

    predictions = []
    prediction_uncertainties = []
    prediction_samples = []

    for b in bands:
        wavelengths = np.ones(len(t_pred)) * get_band_central_wavelength(b)
        pred_x_data = np.vstack([t_pred, wavelengths]).T

        band_pred, band_pred_var = gp(pred_x_data, return_var=True)
        prediction_uncertainties.append(np.sqrt(band_pred_var))

        # s = sample(pred_x_data, 100)

        predictions.append(band_pred)
        # prediction_samples.append(s)


    predictions = np.array(predictions)
    prediction_uncertainties = np.array(prediction_uncertainties)

    return predictions, prediction_uncertainties, prediction_samples

def predict_gaussian_process_new(gp_observations, bands, t_pred, fitted_gp=None, fitted_sample = None, prior = False):
    """
    Parameters
    ----------
    gp_observations: dataframe
    t_pred: dataframe

    Returns
    -------
    predictions:
    prediction_uncertainties:
    
    
    """

    if fitted_gp is not None:
        gp = fitted_gp
    else:
        gp, _, _ = fit_gaussian_process(gp_observations, prior = prior)

    predictions = []
    prediction_uncertainties = []

    for b in bands:
        mask = t_pred["band"] == b
        t_pred_b = t_pred[mask]['relative_time']

        wavelengths = np.ones(len(t_pred_b)) * get_band_central_wavelength(b)
        pred_x_data = np.vstack([t_pred_b, wavelengths]).T

        band_pred, band_pred_var = gp(pred_x_data, return_var=True)
        
        prediction_uncertainties.append(np.sqrt(band_pred_var))
        predictions.append(band_pred)

    return predictions, prediction_uncertainties


def get_band_central_wavelength(band):
    """return the central wavelength of the band"""
    band_central_wavelengths = {'g':4767, 'r': 6215, 'tess': 7865}
    return band_central_wavelengths[band]

def get_band_plot_color(band):
    """Return the plot color for a given band."""
    band_plot_colors = {'g': 'g', 'r': 'r', 'tess':'k'}
    return band_plot_colors[band]

def get_band_plot_marker(band):
    band_plot_markers = {'g': 'o', 'r': 'v', 'tess': '^'}
    return band_plot_markers[band]

def get_band_name(band):
    band_name = {'g': 'g', 'r': 'r', 'tess': 'TESS'}
    return band_name[band]


    
def plot_light_curve(observations, final_filename, axis, prior = False, peak_time_file = None, bands = bands, plot_samples = True, plot_uncert = True, save = False):
    
    gaussian_process, partial_sample, gp_fit_parameters = fit_gaussian_process(observations, prior = prior)
    
    # Figure out the times to plot. We go 10% past the edges of the bservations.
    min_time_obs = np.min(observations['relative_time'])
    max_time_obs = np.max(observations['relative_time'])
    border = 0.1 * (max_time_obs - min_time_obs)
    min_time = min_time_obs - border
    max_time = max_time_obs + border

    t_pred = np.arange(min_time, max_time + 1, step = 1)
    predictions, prediction_uncertainties, prediction_samples = predict_gaussian_process(observations, bands, t_pred, fitted_gp=gaussian_process, fitted_sample=partial_sample, prior = prior)

    # plot mean and standard deviation for peak time
    # peak_time = time_of_peak(final_filename.split('_')[0], prediction_samples, t_pred, df = peak_time_file)
    # if peak_time is not None:
    #     axis.axvline(x =peak_time[0,0])
    #     axis.axvspan(peak_time[0,0] - peak_time[0,1], peak_time[0,0] + peak_time[0,1], alpha = 0.5)
    #     axis.axvline(x =peak_time[0,2], color = 'tab:orange')

    unsorted_bands = np.unique(observations['band'])
    sorted_bands = np.array(sorted(unsorted_bands, key=get_band_central_wavelength))

    for band_idx, b in enumerate(sorted_bands):
        mask = observations["band"] == b
        band_data = observations[mask]
        color = get_band_plot_color(b)
        marker = get_band_plot_marker(b)

        axis.errorbar(
            band_data["relative_time"],
            band_data["flux"],
            band_data["uncert"],
            fmt="o",
            c=color,
            markersize=6,
            marker=marker,
            label=get_band_name(b),
            alpha = 0.4
        )

        pred = predictions[band_idx]
        axis.plot(t_pred, pred, c=color)
        pred_err = prediction_uncertainties[band_idx]
        # pred_samples = prediction_samples[band_idx]

        # if plot_samples:
        #     # plot samples
        #     for s in pred_samples:
        #         axis.plot(t_pred, s, c = color, alpha = 0.1)
        if plot_uncert:
            # show uncertainties with a shaded band.
            axis.fill_between(t_pred, pred - pred_err, pred + pred_err, alpha=0.1, color=color)
    
    axis.legend()

    axis.set_xlabel("Time")
    axis.set_ylabel("Flux")
    axis.set_xlim(min_time, max_time)

    # add bounds for y
    # min_flux_obs = np.min(pred_samples)
    # max_flux_obs = np.max(pred_samples)
    # flux_border = 0.2 * (max_flux_obs - min_flux_obs)
    # min_flux = min_flux_obs - flux_border
    # max_flux = max_flux_obs + flux_border
    # axis.set_ylim(min_flux, max_flux)

    textstr = '\n'.join((
    'parameters: '+ str(gp_fit_parameters),
    'time scale: ' + str(np.sqrt(np.exp(gp_fit_parameters[1])))))

    axis.text(0.05, 0.95, textstr, transform=axis.transAxes, fontsize=10, verticalalignment='top')
    axis.set_title(final_filename[:-3])
    axis.figure.tight_layout()


    if save == True:
        figure.savefig(os.path.join('plots_GP_good_great_notbinned', final_filename))
    else:
        plt.show()

def time_of_peak(id, samples, t_pred, df = None):
    peak_time_constraint = [-20, 25]

    buffer = [t_pred.tolist().index(x) for x in t_pred if x > peak_time_constraint[0] and x < peak_time_constraint[1]]
    if len(buffer) == 0:
        df.loc[len(df.index)] = [id, np.nan, np.nan, np.nan]
        return None
    constrained_t = t_pred[min(buffer):max(buffer) + 1]

    result = []
    time_maxes = []
    for s in samples:
        index_maxes = [x.tolist().index(max(x)) for x in s[:, min(buffer):max(buffer) + 1]]           # get 100 max values from the samples
        time_maxes += [constrained_t[max] for max in index_maxes]
    

    mean = sum(time_maxes)/len(time_maxes)
    median = np.median(time_maxes)
    stdev = statistics.pstdev(time_maxes)

    result.append([mean, stdev, median])            # order: g, r, tess

    if df is not None:
        df.loc[len(df.index)] = [id, mean, stdev, median]

    return np.array(result)





if __name__ == "__main__":
    # matplotlib.use('Agg')

    data_folder = Path('processed_curves_good_great_notbinned')
    parameters = [[],[]]
    # peak_time_file = pd.DataFrame(columns=['object_id', 'mean_time_of_max', 'uncertainty', 'median_time_of_max'])
    figure, axis = plt.subplots()
    
    
    # for testing use
    tess_obj_name = '2018_lit'
    filename = data_folder / 'lc_2018lit_ZTF18adbczrq_processed.csv'
    obs = get_data(filename)

    plot_light_curve(obs, tess_obj_name + '_GP', axis, save = False)
    # fit_gaussian_process(obs, guess_length_scale = 10, prior = False)

    # mass produce plots
    # for file in os.listdir(dir):
    #     tess_obj_name = file.split('_')[1]
    #     obs = get_data(os.path.join(dir, file))

    #     plot_light_curve(obs, tess_obj_name + '_GP', axis, peak_time_file, save = True)
    #     axis.clear()

    # get parameters
    # for file in os.listdir(dir):
    #     tess_obj_name = file.split('_')[1]
    #     obs = get_data(os.path.join(dir, file))
        
    #     _,_,param = fit_gaussian_process(obs)
    #     for i in range(2):
    #         parameters[i].append(param[i])

    # parameters = np.array(parameters)
    # np.save('params_prior.npy', parameters)

    # # load parameters
    # parameters = np.load('params.npy', allow_pickle = True)


    # peak_time_file = peak_time_file.to_csv('plots_peak_good_great/time_of_peak.csv', index=False)




    # d = abs(parameters[1] - np.median(parameters[1]))
    # mad = np.median(d)
    # remove_indexes = parameters[1] > 4 # d < 1000*mad
    # params_removed_outliers = parameters[:,remove_indexes]

    # print(np.mean(params_removed_outliers, axis = 1))
    # print(np.cov(params_removed_outliers))
    # print(np.linalg.inv(np.cov(params_removed_outliers)))

    # means = np.mean(params_removed_outliers, axis = 1)
    # cov = np.cov(params_removed_outliers)
    # npoints = len(parameters[0])
    # samples =  np.random.multivariate_normal(means, cov, npoints)
    # data = np.array(params_removed_outliers).T

    # names = ['param_name',
    #         'const']
            
    # gtc = pygtc.plotGTC(chains = [data, samples])


    # fig = corner.corner(samples);
    # corner.corner(data, fig = fig)


    # plt.show()


