import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C

from CityGeneration.src.dataset.city_dataloader import get_city_dataloader

class ConditionalSampler:
    
    def __init__(self, gp, prior_probs, repel):
        self.gp = gp
        self.repel = repel
        
        self.prior_probs = prior_probs
        
        self.reset()
        
    def reset(self):
        self.X_sampled = np.empty((0, 2))
        self.y_sampled = np.empty((0, 1))
        
    def sample(self, n_samples):
        for _ in range(n_samples):
            # self.gp.fit(self.X_sampled, self.y_sampled)
            
            # x = np.arange(0, self.prior_probs.shape[0])
            # y = np.arange(0, self.prior_probs.shape[1])
            # X, Y = np.meshgrid(x, y)
            # pos = np.dstack((X, Y))
            
            # 

            
            # Predict the GP adjustment on the grid
            # if self.X_sampled.size > 0:
            #     mu_adjustment, _ = self.gp.predict(pos.reshape(-1, 2), return_std=True)
            #     mu_adjustment = mu_adjustment.reshape(self.prior_probs.shape)
            # else:
            #     mu_adjustment = np.zeros_like(self.prior_probs)

            # # Calculate the posterior by combining the prior with GP adjustment
            # mu_adjustment = mu_adjustment * 1
            # posterior_probs = self.prior_probs**2 * np.exp(mu_adjustment)
            # posterior_probs /= posterior_probs.sum()

            # # Sample a new point from the posterior distribution
            # flat_posterior = posterior_probs.ravel()
            # flat_pos = pos.reshape(-1, 2)
            # sampled_idx = np.random.choice(len(flat_posterior), p=flat_posterior)
            # # sampled_idx = np.argmax(flat_posterior)
            # x_new = flat_pos[sampled_idx].reshape(1, -1)
            # y_new = np.array([[-self.repel]])  # "Repel" around this sampled point by setting low GP value

            likelihood = np.zeros_like(self.prior_probs)

            repel_size = 10
            repel_strength = 100
            for point in self.X_sampled:
                heat = np.exp(-((np.arange(self.prior_probs.shape[0]) - point[0]) ** 2 + 
                                (np.arange(self.prior_probs.shape[1])[:,None] - point[1]) ** 2) /
                                (2 * repel_size ** 2))
                
                heat[self.prior_probs == 0] = 0
                heat += 1e-10
                likelihood += heat / heat.sum() 
                
            likelihood = np.exp(-likelihood*repel_strength)
                
            posterior_probs = self.prior_probs * likelihood
            posterior_probs /= posterior_probs.sum()
            
            
            # Sample using the posterior
            flat_posterior = posterior_probs.ravel()
            flat_pos = np.array([[j, i] for i in range(posterior_probs.shape[0]) for j in range(posterior_probs.shape[1])])
            sampled_idx = np.random.choice(len(flat_posterior), p=flat_posterior)
            
            x_new = flat_pos[sampled_idx].reshape(1, -1)
            y_new = np.array([[-self.repel]])  # "Repel" around this sampled point by setting low GP value
            
            
            
            # import matplotlib.pyplot as plt
            # import matplotlib
            # matplotlib.use('TkAgg')
            
            # print(len(self.X_sampled))
            # fig, axs = plt.subplots(1, 3, figsize=(10, 5))
            # axs[0].imshow(self.prior_probs)
            # axs[0].scatter(x_new[:, 0], x_new[:, 1], color='red')
            # axs[0].axis('off')
            # axs[1].imshow(posterior_probs)
            # axs[1].axis('off')
            # axs[2].imshow(likelihood)
            # axs[2].axis('off')
            # plt.show()
            
            # import pdb; pdb.set_trace()
            # # fig, axs = plt.subplots(1, 2, figsize=(10, 5))
            # # axs[0].imshow(posterior_probs)
            # # axs[0].scatter(x_new[:, 0], x_new[:, 1], color='red')
            
            # # if self.X_sampled.shape != (1, 0):
            # #     axs[0].scatter(self.X_sampled[:, 0], self.X_sampled[:, 1], color='blue')
            # # axs[0].axis('off')
            # # axs[1].imshow(mu_adjustment)
            # # axs[1].axis('off')
            
            # plt.show()
            
            # matplotlib.use('Agg')
            
            

            # Update GP with the new sample point
            if self.X_sampled.shape == (1, 0):
                self.X_sampled = x_new
                self.y_sampled = y_new
            else:
                self.X_sampled = np.vstack([self.X_sampled, x_new])
                self.y_sampled = np.vstack([self.y_sampled, y_new])
            
            
            
            
            
            # plot the heatmap
            # fig, axs = plt.subplots(1, 4, figsize=(10, 5))
            # vmin = min(self.prior_probs.min(), posterior_probs.min(), (self.prior_probs - posterior_probs).min())
            # vmax = max(self.prior_probs.max(), posterior_probs.max(), (self.prior_probs - posterior_probs).max())
            
            # axs[0].imshow(self.prior_probs, vmin=vmin, vmax=vmax)
            # axs[0].scatter(self.X_sampled[:, 0], self.X_sampled[:, 1], color='red')
            # axs[0].axis('off')
            # axs[1].imshow(posterior_probs, vmin=vmin, vmax=vmax)
            # axs[1].axis('off')
            # axs[2].imshow(self.prior_probs - posterior_probs)
            # axs[2].axis('off')
            # axs[3].imshow(mu_adjustment)
            # plt.show()

if __name__ == "__main__":
    
    import matplotlib.pyplot as plt
    import yaml
    
    with open("CityGeneration/config/config.yaml", "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    dataloader, dataset = get_city_dataloader(data_kwargs=config["dataset"]["params"])
    
    for i in range(len(dataset)):
        
        img, heatmap = dataset[i]
    
        # 1. Define the kernel and initialize the GP
        # use the heatmap as the mean function
        kernel = RBF(5, (1e-10, 1e6))
        gp = GaussianProcessRegressor(kernel=kernel, 
                                      optimizer = None,
                                      alpha=1e-6, n_restarts_optimizer=0)
        
        sampler = ConditionalSampler(gp, 
                                     heatmap[0].numpy(),
                                     repel = 0.5)
        
        
        sampler.sample(10)
        
        fig, axs = plt.subplots(1, 3, figsize=(10, 5))
        
        axs[0].imshow(img.permute(1, 2, 0))
        axs[0].scatter(sampler.X_sampled[:, 0], sampler.X_sampled[:, 1], color='red')
        axs[0].axis('off')
        axs[1].imshow(heatmap.permute(1, 2, 0))
        axs[1].scatter(sampler.X_sampled[:, 0], sampler.X_sampled[:, 1], color='red')
        axs[1].axis('off')
        axs[2].imshow(heatmap.permute(1, 2, 0)**2)
        axs[2].scatter(sampler.X_sampled[:, 0], sampler.X_sampled[:, 1], color='red')
        axs[2].axis('off')
        plt.show()
        