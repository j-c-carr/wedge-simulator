from tqdm import tqdm
import typing
from typing import Optional, List
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors

eor_colour = colors.LinearSegmentedColormap.from_list(
    "EoR",
    [
        (0, "white"),
        (0.21, "yellow"),
        (0.42, "orange"),
        (0.63, "red"),
        (0.86, "black"),
        (0.9, "blue"),
        (1, "cyan"),
    ],
)
plt.register_cmap(cmap=eor_colour)
#plt.style.use("lightcones/plot_styles.mplstyle")

COLORS = plt.rcParams["axes.prop_cycle"].by_key()["color"]
#print(COLORS)

class LightconePlotManager():

    """Manager class for lightcone plotting functions."""

    def __init__(self, 
                 lightcone_redshifts: np.ndarray,
                 lightcone_shape: tuple,
                 lightcone_dimensions: tuple,
                 los_axis:int = 0, 
                 mpc_axis: int = 1) -> None:
        """
        Params:
        :lightcone_redshifts: list of redshifts along the LoS
        axis.
        :lightcone_shape: shape of the lightcone in pixels. LoS axis
        MUST be the first dimension.
        :lightcone_dimensions: physical dimensions of the lightcone in
        Mpc. LoS axis MUST be the first dimension.
        :los_axis: line of sight axis of the lightcone. This is constant
        for each dataset and so each instance of the LightconePlotManager
        should have the same los_axis.
        :mpc_axis: any axis other than the los_axis
        """

        assert los_axis != mpc_axis
        assert max(los_axis, mpc_axis) < len(lightcone_shape)
        assert lightcone_redshifts.shape[0] == lightcone_shape[los_axis]

        self.lightcone_redshifts = lightcone_redshifts
        self.lightcone_shape = lightcone_shape
        self.lightcone_dimensions = lightcone_dimensions
        self.los_axis = los_axis
        self.mpc_axis = mpc_axis

        self.redshift_labels = [
                round(self.lightcone_redshifts[i], 2) for i in
                    range(0, self.lightcone_redshifts.shape[0],
                        self.lightcone_redshifts.shape[0]//(3*lightcone_shape[0]//128))]

        self.redshift_ticks = np.arange(0, lightcone_shape[los_axis],
                step=lightcone_shape[los_axis]/len(self.redshift_labels))

        self.mpc_labels = np.arange(0, lightcone_dimensions[mpc_axis],
                step=lightcone_dimensions[mpc_axis]//6)

        self.mpc_ticks = np.arange(0, lightcone_shape[mpc_axis], 
                lightcone_shape[mpc_axis]/len(self.mpc_labels))

        # Default matplotlib args
        self.kwargs = {
                "origin": "lower",
                "aspect": "auto",
                "cmap": "coolwarm"
                }

        self.width_ratios = [1,
                lightcone_shape[los_axis]//lightcone_shape[mpc_axis]]


    def set_ticks_and_labels(self,
                             ax: List[plt.Axes],
                             labels: List[str],
                             n: int) -> None:
        """Set the ticks and labels for the axes"""
        for i in range(n):
            ax[i][0].set_xticks(self.mpc_ticks)
            ax[i][0].set_xticklabels([])
            ax[i][0].set_yticks(self.mpc_ticks)
            ax[i][0].set_yticklabels([])
            ax[i][0].set_ylabel(labels[i])

            ax[i][1].set_xticks(self.redshift_ticks)
            ax[i][1].set_xticklabels([])
            ax[i][1].set_yticks(self.mpc_ticks)
            ax[i][1].set_yticklabels([])
            # ax[i][1].set_ylabel(labels[i])


        # Put labels on last plots only
        ax[n-1][0].set_xticklabels(self.mpc_labels)
        ax[n-1][0].set_xlabel("Mpc")
        ax[n-1][1].set_xticklabels(self.redshift_labels)
        ax[n-1][1].set_xlabel(r"$z$")


    def compare_lightcones(self, 
                           prefix: str,
                           L: dict,
                           D: Optional[dict] = None,
                           T: Optional[List[str]] = None,
                           num_samples: int = 1) -> None:
        """
        Wrapper function for plot_lightcones to plot multiple lightcones and
        save them. Used for comparing model predictions and ground truth.

        Parameters:
        -----------
        :prefix: prefix of image filename
        :L: dict of n sets of lightcones of shape (batch_size, *lightcone_shape)
        :D: dict of other data to plot of form: {data_label: [X, Y]}
        :T: Data for table plot
        :num_samples: number of plots to make.
        """

        # Plot random samples
        I = np.random.randint(list(L.values())[0].shape[0], size=num_samples)
        for i in I:
            if D is None:
                self.plot_lightcones(np.array([l[i, ...] for l in L.values()]),
                                    list(L.keys()))
            else:
                self.plot_lightcones(np.array([l[i, ...] for l in L.values()]),
                                    list(L.keys()), D=D)


            plt.tight_layout()
            plt.savefig(f"{prefix}_{i}.png", dpi=400)
            plt.close()


    def plot_lightcones(self, 
                        L: np.ndarray,
                        labels: List[str],
                        mpc_slice_index: Optional[int] = None,
                        los_slice_index: Optional[int] = None,
                        D: Optional[dict] = None,
                        kwargs: Optional[dict] = None) -> None:
        """
        Plots a transverse and LoS slice of each lightcone in L. 
        -----
        Params:
        :L: list of of n lightcones of shape (n, *lightcone_shape)
        :labels: list of of n labels
        :slice_index: (int) slice index for plotting
        :D: dictionary of extra data to plot of form {data_label: [X,Y]}
        :kwargs: (dict) matplotlib parameters
        """
        
        if kwargs is None:
            kwargs = self.kwargs

        if mpc_slice_index is None:
            mpc_slice_index = np.random.randint(0, self.lightcone_shape[self.los_axis])
            
        if los_slice_index is None:
            los_slice_index = np.random.randint(0, self.lightcone_shape[self.mpc_axis])

        assert mpc_slice_index < self.lightcone_shape[self.los_axis] 
        assert los_slice_index < self.lightcone_shape[self.mpc_axis] 

        # Make the figure bigger if there are extra plots
        if D is None:
            N = L.shape[0]
        else:
            N = L.shape[0] + len(D.keys())

        fig, ax = plt.subplots(nrows=N, ncols=2, 
                               figsize=((self.width_ratios[1]+1)*2, N*2),
                               gridspec_kw={'width_ratios': self.width_ratios})
        
        # Plot tranverse and los slice for each lightcone
        for i in range(L.shape[0]):

            # Transverse slice plot
            trans_slice = np.take(L[i], mpc_slice_index, axis=self.los_axis)
            pos0 = ax[i][0].imshow(trans_slice, **kwargs)
            cbar0 = fig.colorbar(pos0, ax=ax[i][0], pad=0.01)
            
            # LoS slice plot + colorbar
            los_slice = np.take(L[i], los_slice_index, axis=self.mpc_axis).T
            pos1 = ax[i][1].imshow(los_slice, **kwargs)
            cbar1 = fig.colorbar(pos1, ax=ax[i][1], pad=0.01)



        # Plot extra data
        if D is not None:

            assert list(D.values())[0][1].shape[0] == L.shape[1], \
                        f"expected shape {L.shape[1]}," \
                        f"got shape {list(D.values())[0].shape[1]}"

            i = L.shape[0]
            for label, data in D.items():
                ax[i][0].scatter(*data)
                ax[i][1].scatter(*data)

                ax[i][1].set_xlim(0, L.shape[1])
                ax[i][1].set_xticks(self.redshift_ticks)
                ax[i][1].set_xticklabels(self.redshift_labels)
                ax[i][1].set_ylabel(label)
                i+=1

        ax[0][0].set_title("$\Delta T$ (Trans), $z=$ {:.2f}".format(
                                        self.lightcone_redshifts[mpc_slice_index]))
        ax[0][1].set_title(r"$\Delta T$ (LoS)")
        self.set_ticks_and_labels(ax, labels, L.shape[0])


    def lightcone_movie(self, masks, redshifts):


        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(6,3), dpi=300)
        camera = Camera(fig)

        for _, mask in enumerate(tqdm(masks)):
            ax.imshow(mask.T, **self.kwargs)
            ax.set_xticks(self.redshift_ticks)
            ax.set_xticklabels(self.redshift_labels)
            ax.set_yticks(self.mpc_ticks)
            ax.set_yticklabels([])
            plt.tight_layout()

            camera.snap()



        movie = camera.animate(interval=30)
        movie.save("rolling-wedge-movie.mp4")

