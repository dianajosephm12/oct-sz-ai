import matplotlib.pyplot as plt
import numpy as np

e1 = 'intermediates/e1_[1L, 64L, 256L, 248L]_049.npy'
e1_conv = np.load('intermediates/e1'.format(e1=e1))
plt.imshow(e1_conv[0, :, :, 200])
plt.show()

# npy_file = 'e3_[1L, 64L, 64L, 62L]_049.npy'
# # npy_file = 'out_encoder_[1L, 64L, 64L, 62L]_049.npy'
# conv = np.load('intermediates/npy_file'.format(npy_file=npy_file))
# print(conv.shape)
# images_per_row = 8
# size = conv.shape[1]
# n_features = conv.shape[-1] - 50    # subtracted 50 for visualization sake
# n_cols = n_features // images_per_row
# display_grid = np.zeros((size * n_cols, images_per_row * size))
# for col in range(n_cols): # Tiles each filter into a big horizontal grid
#     for row in range(images_per_row):
#         channel_image = conv[0, :, :, col * images_per_row + row]
#         channel_image -= channel_image.mean() # Post-processes the feature to make it visually palatable
#         channel_image /= channel_image.std()
#         channel_image *= 64
#         channel_image += 128
#         channel_image = np.clip(channel_image, 0, 255).astype('uint8')
#         display_grid[col * size : (col + 1) * size, # Displays the grid
#                      row * size : (row + 1) * size] = channel_image
# scale = 1. / size
# plt.figure(figsize=(scale * display_grid.shape[1],
#                     scale * display_grid.shape[0]))
# plt.title(npy_file)
# plt.grid(False)
# plt.imshow(display_grid, aspect='auto', cmap='viridis')
#
# # plt.imshow(conv[0, :, :, 5])
# plt.show()