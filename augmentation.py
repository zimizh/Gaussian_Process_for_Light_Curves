import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from GP_time_wavelength import get_data, fit_gaussian_process, predict_gaussian_process, predict_gaussian_process_new, get_band_central_wavelength, get_band_plot_color, get_band_plot_marker, get_band_name
import time

# Plot three histograms (g, r, tess) of the number of observations in each light curve
# Create a list of all observations times. Create a list of nobs number of random passbands
# Use your GP to predict at each of these times and passbands. (at the GP mean)
# Plot histogram of flux uncertainties in each band
# Fit a log-normal to that histogram
# Pick random number from the above distribution

#######

class Augmentation(object):
    """ augment data"""
    figure, axis = plt.subplots()

    def __init__(self, filename, obj_name, obs):
        self.filename = filename
        self.obj_name = obj_name
        self.obs = obs

    def bands(self):
        """Return a list of bands that this object has observations in
        Returns
        -------
        bands : numpy.array
            A list of bands, ordered by their central wavelength.
        """
        unsorted_bands = np.unique(self.obs["band"])
        sorted_bands = np.array(sorted(unsorted_bands, key=get_band_central_wavelength))
        return sorted_bands
    
    def choose_sampling_times(self, params):
        """step 2"""
        t_obs = self.obs[['relative_time', 'band']]
        bands = self.bands()
      
        # range of observation time for the original data
        t_range = {band:[] for band in bands}
        for b in bands:
            mask = t_obs["band"] == b
            b_times = t_obs[mask]
            
            t_range[b].append(b_times.iloc[0]['relative_time'])
            t_range[b].append(b_times.iloc[-1]['relative_time'])

        # target observation count not binned
        tgt_obs = {'g':np.array([1.6339911154483238, 0.3688840047158963]),
                   'r':np.array([1.649320766643334, 0.38343472939314077]),
                   'tess':np.array([[995.892777, 2913.391893],[106.448989,256.318191],[0.6636387663746808, 0.3363612336253193]])
                   }

        nsamp_bands, samp_t, samp_t_bands = np.zeros(np.shape(bands)), np.zeros(np.shape(bands)), np.zeros(np.shape(bands))
        
        def sample_obs_count(band_name):
            """
            given a band, return the sample observational count
            """
            if band_name == 'g' or band_name == 'r':
                return np.round(pow(10, np.random.normal(tgt_obs[band_name][0], tgt_obs[band_name][1])))
            elif band_name == 'tess':
                tess_mixture = GaussianMixture(n_components = 2, covariance_type = 'spherical')
                tess_mixture.means_ = tgt_obs['tess'][0].reshape((2, -1))
                tess_mixture.covariances_ = tgt_obs['tess'][1]
                tess_mixture.weights_ = tgt_obs['tess'][2]
                return np.round(tess_mixture.sample(1)[0][0][0])
            
        for i, band in enumerate(bands):
            nsamp_bands[i] = sample_obs_count(band)
        
        nsamp_tot = np.sum(nsamp_bands)

        b_sample_list = np.random.choice(bands, int(nsamp_tot), p = nsamp_bands/nsamp_tot)                  # sample a list of passbands e.g. [r g r r tess g tess tess g r]
        
        def resample(sampl_t, t_range, t_obs, band, params):
            """
            resample until the sample is unique and within the time range of the original data

            if params are at the boundary, only sample r and g bands from their own bands
            """
            loop_count = 0
            
            while True:
                unique, repeat_counts = np.unique(sampl_t, return_counts=True)

                min = t_range[band][0]   
                max = t_range[band][1]
                unique_in_range = unique[((unique <= max) & (unique >= min))] 

                if len(unique_in_range) == len(sampl_t):
                    return sampl_t
                
                if band == 'tess' or loop_count > 10:
                    repeated_times = unique[(repeat_counts > 1)]
                    jitter_times = np.random.uniform(-0.0104167, 0.0104167, size=len(repeated_times))
                    new_times = repeated_times + jitter_times
                else:
                    new_times = np.random.choice(t_obs['relative_time'][((t_obs['band'] == 'g') | (t_obs['band'] == 'r'))], len(sampl_t) - len(unique_in_range), replace = True)

                sampl_t = np.concatenate([unique_in_range, new_times])

                loop_count += 1
        
        for i, band in enumerate(bands):
            if band == 'tess':
                # WAIT why can't we just directly generate the times for each passband???
                samp_t[i] = np.random.choice(t_obs['relative_time'], int(nsamp_tot), replace = True)                                        # sample nsamp_tot TESS times from all of the times
            else:
                # need to check parameters here
                samp_t[i] = np.random.choice(t_obs['relative_time'][(t_obs['band'] == band)], int(nsamp_tot), replace = True)               # sample nsamp_tot g and r times

            samp_t_bands[i] = samp_t[i][b_sample_list == band]                                                                  # overlay the sample times with list of sample passbands to choose times for each passband
            samp_t_bands[i] = resample(samp_t_bands[i], t_range, t_obs, band, params)                                           # resample until unique and within bounds


        bands = ['g']*len(g_sampl_t) + ['r']*len(r_sampl_t) + ['tess']*len(tess_sampl_t)#aaaaaaaahhhhhhhh
        data = {'relative_time': np.flatten(samp_t_bands),
                'band': bands}
       
        augmented_lc = pd.DataFrame(data)

        return augmented_lc

    def choose_flux_uncert(self, sampl_t, predictions, prediction_uncert):
        # target flux uncertainties
        mean = [3.614981032804085, 3.6999864555829736, 245.56637449519062]
        std = [0.49336796312463577, 0.4707468394178932, 73.47659776298671]

        new_predictions = []
        new_uncert = []
        
        for i, fluxunc_pb in enumerate(prediction_uncert):
            if i == 0 or i == 1:
                sigma_d = np.random.lognormal(mean[i], std[i], len(fluxunc_pb))
            else:
                sigma_d = np.random.normal(mean[i], std[i], len(fluxunc_pb))

            sigma_aug = np.sqrt(np.add(np.square(np.array(fluxunc_pb, dtype = np.float64)), np.square(np.array(sigma_d, dtype = np.float64))))
            
            predictions_moved = (np.random.normal(predictions[i], sigma_aug))

            new_predictions.append(predictions_moved)
            new_uncert.append(sigma_aug)
  
        return new_predictions, new_uncert

    def gp_predict(self, regular_interval = False):
        """step 3"""
        gp,_,params = fit_gaussian_process(self.obs)

        aug_lc = self.choose_sampling_times(params)

        if regular_interval:
            predictions, prediction_uncertainties = predict_gaussian_process(self.obs, self.bands(), aug_lc, fitted_gp = gp)
        else:
            predictions, prediction_uncertainties = predict_gaussian_process_new(self.obs, self.bands(), aug_lc, fitted_gp = gp)

        new_predictions, new_uncert = self.choose_flux_uncert(aug_lc, predictions, prediction_uncertainties)
        
        aug_lc['flux'] = [x for sublist in new_predictions for x in sublist]
        aug_lc['uncert'] = [x for sublist in new_uncert for x in sublist]
        
        return aug_lc


    def augment_curve(self, save_csv, save_graph, regular_interval = False, foldername = None, filename = None):
        """plotting the augmented curve"""
        aug_lc = self.gp_predict(regular_interval)

        if save_csv:
            aug_lc.to_csv(os.path.join('augmented_lc', filename), index = False)

        self.plot_curve(aug_lc, save_graph, foldername, filename)

    def plot_original_curve(self, save, foldername, filename):
        self.plot_curve(self.obs, save, foldername, filename)

    def plot_curve(self, observations, save, foldername = None, filename = None):
       
        for band_idx, band in enumerate(self.bands()):
            mask = observations["band"] == band
            band_data = observations[mask]
            color = get_band_plot_color(band)
            marker = get_band_plot_marker(band)
            
            self.axis.errorbar(
                    band_data["relative_time"],
                    band_data["flux"],
                    band_data["uncert"],
                    fmt="o",
                    c=color,
                    markersize=6,
                    marker=marker,
                    label=get_band_name(band),
                    alpha = 0.4
                )
        self.axis.legend()

        self.axis.set_xlabel("Time")
        self.axis.set_ylabel("Flux")
        
        self.axis.set_title(filename)
        self.axis.figure.tight_layout()

        if save == True:
            self.figure.savefig(os.path.join(foldername, filename))    
        else:
            plt.show()

        plt.cla()


if __name__ == '__main__':  

    # repeat = 20

    # dir = 'processed_curves_good_great_notbinned'
    # i = os.listdir(dir).index('lc_2019axj_ZTF19aajwjwq_processed.csv')
    # for file in os.listdir(dir)[i:]:
    #     tess_obj_name = file.split('_')[1]
    #     filename = os.path.join(dir, file)

    #     obs = get_data(filename)

    #     augment = Augmentation(filename, tess_obj_name, obs)
        
    #     for i in range(repeat):
    #         augment.augment_curve(save_csv = True, save_graph = True, regular_interval = False, foldername = 'plots_augmented_lc', filename = tess_obj_name + '_augmented_' + str(i))

    
    
    tess_obj_name = '2019bip'
    filename = 'processed_curves_good_great_notbinned\lc_2019bip_ZTF19aallimd_processed.csv'

    obs = get_data(filename)
    augment = Augmentation(filename, tess_obj_name, obs)

    augment.augment_curve(save_csv = False, save_graph = False)