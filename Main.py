"""Minimal runner for the refactored package."""
from src.cmb.model import CosmologyAdvancedSampling


def main():
    model = CosmologyAdvancedSampling(_lmax=8, _NSIDE=2, _noisesig=1.0)
    print("Constructed CosmologyAdvancedSampling with:")
    print(f" lmax={model.lmax}, NSIDE={model.NSIDE}, NPIX={model.NPIX}")


if __name__ == "__main__":
    main()
        #__psi_record3.append(tf.reduce_sum(_psi3).numpy())
        #__psi_record.append(_psi.numpy())
        #print('psi1',tf.reduce_sum(_psi1),'psi2',tf.reduce_sum(_psi2),'psi3',tf.reduce_sum(_psi3))
        return _psi

