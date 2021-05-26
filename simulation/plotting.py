import os
import glob
import numpy as np
from pathlib import Path
from scipy.stats import pearsonr
from matplotlib import pyplot as plt 


# Scatter cluster profiles against each other 
def scatter_profiles(cluster_profiles, selected_profiles):
    nct = len(selected_profiles)

    fig = plt.figure(figsize=(3*nct,3*nct))
    for i in range(nct):
        for j in range(i, nct):
            ax = plt.subplot(nct, nct, i*nct + j + 1)
            
            ax.axline([0, 0], [1, 1], c='r')
            
            x = cluster_profiles[selected_profiles[i]].values
            y = cluster_profiles[selected_profiles[j]].values
            ax.scatter(x, y, s=1)

            r, p = pearsonr(x, y)
            plt.text(500, 1.5, 'r = %.3f' % r, fontsize=10)

            ax.set_xlim(1,1e4)
            ax.set_ylim(1,1e4)
            ax.set_aspect('equal')

            ax.set_xscale('log')
            ax.set_yscale('log')

            ax.set_xlabel(selected_profiles[i] + ' log(TPM+1)', fontsize=8)
            ax.set_ylabel(selected_profiles[j] + ' log(TPM+1)', fontsize=8)
    plt.tight_layout()
    return fig


# Scatter recovered profiles (mean exponentiated cSplotch Betas) against cluster profiles
def scatter_recovery(recovery_dir, cluster_profiles, aar_profiles):
    lambda_mu, lambda_sig, true_tpm = [],[], []
    aar_profiles = np.array(aar_profiles)
    
    # Parse mean lambdas for all genes and match with true TPM values
    for mu_fn in glob.glob(os.path.join(recovery_dir, 'lambda_*_mu.npy')):
        sig_fn = mu_fn.replace('mu', 'sigma')    
        lambda_mu.append(np.load(mu_fn)[0])   # Remove condition dimension 
        lambda_sig.append(np.load(sig_fn)[0]) 
        
        gid = int(Path(mu_fn).stem.split('_')[1])
        
        if aar_profiles.ndim == 1:
            true_tpm.append(np.expand_dims(cluster_profiles[aar_profiles].values[gid,:], 0))
        elif aar_profiles.ndim == 2:
            true_tpm.append(np.vstack([cluster_profiles[aap].values[gid,:] for aap in aar_profiles]))
        else:
            raise ValueError('aar_profiles should be of dimension (n_aar, n_celltype)')
                
    lambda_mu = np.array(lambda_mu)
    lambda_sig = np.array(lambda_sig)
    true_tpm = np.array(true_tpm)
            
    _, n_aar, n_ct = lambda_mu.shape
    fig = plt.figure(figsize=(3*n_ct, 3*n_aar))
    
    for a in range(n_aar):
        for i in range(n_ct):
            ax = plt.subplot(n_aar, n_ct, a * n_ct + i + 1)
            
            x = true_tpm[:,a,i]
            y = lambda_mu[:,a,i] * 1e6 + 1
            err = lambda_sig[:,a,i] * 1e6 

            ax.axline([0, 0], [1, 1], c='r')        
            ax.errorbar(x, y, yerr=err, fmt='o', markersize=1)

            r, p = pearsonr(x, y)
            ax.text(500, 1.5, 'r = %.3f' % r, fontsize=10)

            ax.set_xlim(1,1e4)
            ax.set_ylim(1,1e4)

            ax.set_xscale('log')
            ax.set_yscale('log')
            ax.set_aspect('equal')

            ax.set_xlabel('True log(TPM+1)', fontsize=8)
            ax.set_ylabel('Predicted log(TPM+1)', fontsize=8)
            
            if aar_profiles.ndim == 1:
                ax.set_title(aar_profiles[i], fontsize=10)
            else:
                ax.set_title(aar_profiles[a,i], fontsize=10)
    plt.tight_layout()
    return fig


# Scatter recovered cSplotch profiles (mean exponentiated Betas) against each other, as well as those obtained from vanilla Splotch
def scatter_against_splotch(csplotch_recovery_dir, splotch_recovery_dir, aar_profiles):
    c_lambda_mu, s_lambda_mu = [],[]
    aar_profiles = np.array(aar_profiles)

    for c_mu_fn in glob.glob(os.path.join(csplotch_recovery_dir, 'lambda_*_mu.npy')):
        c_lambda_mu.append(np.load(c_mu_fn)[0])  # Remove condition dimension

        s_mu_fn = os.path.join(splotch_recovery_dir, Path(c_mu_fn).name)
        s_lambda_mu.append(np.load(s_mu_fn)[0])

    c_lambda_mu = np.array(c_lambda_mu)
    s_lambda_mu = np.array(s_lambda_mu)
    
    _, n_aar, n_ct = c_lambda_mu.shape

    fig, ax = plt.subplots(n_aar*n_ct, n_aar*n_ct+1, figsize=(3*(n_ct*n_aar+1),3*n_ct*n_aar))
    for i in range(n_aar*n_ct):
        for j in range(n_aar*n_ct+1):
            ax[i,j].axis('off')
    
    for a in range(n_aar):
        for i in range(n_ct):
            for j in range(i, n_ct+1):
                x = c_lambda_mu[:,a,i] * 1e6 + 1
                
                # Comparison against other cSplotch profile
                if j < n_ct:
                    y = c_lambda_mu[:,a,j] * 1e6 + 1
                    
                    plot_row, plot_col = a*n_ct+i, a*n_ct+j
                    color = None
                    
                    if aar_profiles.ndim == 1:
                        ax[plot_row, plot_col].set_ylabel(aar_profiles[j] + ' log(TPM+1)', fontsize=8)
                    else:
                        ax[plot_row, plot_col].set_ylabel(aar_profiles[a,j] + ' log(TPM+1)', fontsize=8)
                    
                # Comparison against vanilla Splotch
                else:
                    y = s_lambda_mu[:,a] * 1e6 + 1
                    
                    plot_row, plot_col = a*n_ct+i, -1
                    color = 'g'
                    
                    ax[plot_row, plot_col].set_ylabel('Splotch log(TPM+1)', fontsize=8)
            
                ax[plot_row, plot_col].axis('on')
                ax[plot_row, plot_col].axline([0, 0], [1, 1], c='r')
                ax[plot_row, plot_col].scatter(x, y, s=1, c=color)

                r, p = pearsonr(x, y)
                ax[plot_row, plot_col].text(500, 1.5, 'r = %.3f' % r, fontsize=10)

                ax[plot_row, plot_col].set_xlim(1,1e4)
                ax[plot_row, plot_col].set_ylim(1,1e4)
                ax[plot_row, plot_col].set_aspect('equal')

                ax[plot_row, plot_col].set_xscale('log')
                ax[plot_row, plot_col].set_yscale('log')

                if aar_profiles.ndim == 1:
                    ax[plot_row, plot_col].set_xlabel(aar_profiles[i] + ' log(TPM+1)', fontsize=8)
                else:
                    ax[plot_row, plot_col].set_xlabel(aar_profiles[a,i] + ' log(TPM+1)', fontsize=8)
    plt.tight_layout()

    return fig
