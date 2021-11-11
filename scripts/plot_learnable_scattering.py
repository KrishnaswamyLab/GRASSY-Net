import matplotlib.pyplot as plt

TRANCH_NAME = 'BBAB'

model = Scatter(10, trainable_laziness=False)
model.load_state_dict(torch.load('./trained_models/' + f"{TRANCH_NAME}.npy"))
model.eval()
wavelet_constructor = model.wavelet_constructor.detach().numpy()

x = []

for i in range(17):
    x.append(i)

plt.figure(figsize=(8,6))
plt.scatter(x, wavelet_constructor[0], label='$\Phi_0$')
plt.scatter(x, wavelet_constructor[1] - wavelet_constructor[2], label='$\Psi_1$')
plt.scatter(x, wavelet_constructor[2] - wavelet_constructor[3], label='$\Psi_2$')
plt.title(TRANCH_NAME)
plt.legend()
plt.savefig(f'./trained_models/plots/{TRANCH_NAME}.png')
