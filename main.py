import matplotlib.pyplot as plt
import numpy as np

# numpyで1行に表示できる文字数を増やす
np.set_printoptions(linewidth=200)


def gcc_phat(sig, refsig, fs=1, max_tau=None, interp=16):
    """
    This function computes the offset between the signal sig and the reference signal refsig
    using the Generalized Cross Correlation - Phase Transform (GCC-PHAT)method.
    """

    # make sure the length for the FFT is larger or equal than len(sig) + len(refsig)
    n = sig.shape[0] + refsig.shape[0]

    # Generalized Cross Correlation Phase Transform
    SIG = np.fft.rfft(sig, n=n)
    REFSIG = np.fft.rfft(refsig, n=n)
    R = SIG * np.conj(REFSIG)

    cc = np.fft.irfft(R / np.abs(R), n=(interp * n))

    max_shift = int(interp * n / 2)
    if max_tau:
        max_shift = np.minimum(int(interp * fs * max_tau), max_shift)

    cc = np.concatenate((cc[-max_shift:], cc[: max_shift + 1]))

    # find max cross correlation index
    shift = np.argmax(np.abs(cc)) - max_shift

    tau = shift / float(interp * fs)

    return tau, cc


def main():
    refsig = np.linspace(1, 10, 10)

    correlation_arrays = []

    for i in range(0, 10):
        sig = np.concatenate((np.linspace(0, 0, i), refsig, np.linspace(0, 0, 10 - i)))
        offset, cc = gcc_phat(sig, refsig)
        print(f"{sig}, {refsig} -> {offset}")
        correlation_arrays.append(cc)

    # Plotting the correlation arrays
    plt.figure(figsize=(15, 10))
    for i, cc in enumerate(correlation_arrays):
        plt.subplot(5, 2, i + 1)
        plt.plot(cc)
        plt.title(f"Correlation with {i} unit delay")
        plt.xlabel("Lag")
        plt.ylabel("Correlation")
    plt.tight_layout()
    plt.savefig("correlation_graphs.png")


if __name__ == "__main__":
    main()
