"""Run the Lanczos iteration on the implicit Hessian of a pretrained model.
"""
from absl import app
from absl import flags
from local_ensembles import run_local_ensembles

flags.DEFINE_string('expname', 'temp', 'name of this experiment directory')
flags.DEFINE_string('expdir', 
        '/scratch/gobi1/madras/extrapolation/jul4_clf_ood_mnist_ood5/', 
        'where is the trained classifier located')
flags.DEFINE_integer('num_examples', 100, 'how many examples we caluclate on')
flags.DEFINE_integer('batch_size', 32, 'batch size for dataset iterators')
flags.DEFINE_integer('num_lanczos_iterations', 100, 
        'how many iterations to run Lanczos for')
flags.DEFINE_integer('projection_step', 1, 
        'sample projections skipping how many?')
flags.DEFINE_string('model_type', 'CNN_classifier', 
        'what type of model to load - CNN_classifier or MLP_regressor')
flags.DEFINE_string('ckpt_name', 'bestmodel', 'name of checkpoint to load')
flags.DEFINE_bool('two_reorth', False, 'should we do reorth 2x at each step?')
flags.DEFINE_integer('tf_seed', None, 'random seed for Tensorflow')
flags.DEFINE_integer('np_seed', None, 'random seed for Numpy')
flags.DEFINE_bool('run_baselines', True, 'run baselines')
flags.DEFINE_bool('use_prediction_gradient', True, 'use prediction gradient'
        ' for calculating extrapolation scores')
flags.DEFINE_integer('n_labels', 1, 'if using loss gradients and MLP_regressor,'
        ' number of labels to loop over')

FLAGS = flags.FLAGS

def main(argv):
    if len(argv) > 1:
        raise app.UsageError('Too many command-line arguments.')
    params = FLAGS.flag_values_dict()
    run_local_ensembles.main(params)

if __name__ == '__main__':
    app.run(main)
