import torch
from torch.autograd import Variable
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter, MaxNLocator
import os
import pickle
import onnx
import onnxruntime as ort
from tqdm import tqdm
from evaluation.metrics import calculate_metrics

class BaseTrainer:
    @staticmethod
    def add_trainer_args(parser):
        """Adds arguments to Paser for training process"""
        parser.add_argument('--lr', default=None, type=float)
        parser.add_argument('--model_file_name', default=None, type=float)
        return parser

    def __init__(self):
        pass

    def train(self, data_loader):
        pass

    def valid(self, data_loader):
        pass

    def test_pth(self, data_loader):
        raise NotImplementedError("test_pth method must be implemented in the subclass")
    
    def test_onnx_batch(self, test_batch, ort_session, predictions, labels):
        raise NotImplementedError("test_onnx_batch method must be implemented in the subclass")
    
    def test_onnx(self, data_loader):
        """Model evaluation using ONNX runtime."""
        if data_loader["test"] is None:
            raise ValueError("No data for test")

        print('')
        print("===Testing with ONNX===")

        # Change chunk length to be test chunk length
        self.chunk_len = self.config.TEST.DATA.PREPROCESS.CHUNK_LENGTH

        # Check if ONNX model exists, if not export it
        if not os.path.exists(self.onnx_path):
            print(f"ONNX model not found at {self.onnx_path}, exporting...")
            self.export_to_onnx()

        # Create ONNX Runtime session
        available_providers = ort.get_available_providers()
        print(f"Available ONNX providers: {available_providers}")
        
        # Select providers based on availability and device configuration
        if 'CUDAExecutionProvider' in available_providers:
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
            print("Using GPU acceleration for ONNX inference")
        else:
            providers = ['CPUExecutionProvider']
            print("Using CPU for ONNX inference")
        
        print(f"Selected ONNX providers: {providers}")
        ort_session = ort.InferenceSession(self.onnx_path, providers=providers)
        
        print(f"Testing uses ONNX model: {self.onnx_path}")
        print("Running model evaluation on the testing dataset using ONNX!")
        
        predictions = dict()
        labels = dict()
        
        with torch.no_grad():
            for _, test_batch in enumerate(tqdm(data_loader["test"], ncols=80)):
                self.test_onnx_batch(test_batch, ort_session, predictions, labels)
                
        calculate_metrics(predictions, labels, self.config)
    
    def test(self, data_loader):
        # self.test_pth(data_loader)
        self.test_onnx(data_loader)
    
    def get_dummy_input(self):
        raise NotImplementedError("get_dummy_input method must be implemented in the subclass")
        
    def export_to_onnx(self):
        """Export the trained model to ONNX format."""
        model_path = self.config.INFERENCE.MODEL_PATH
        
        if model_path and os.path.exists(model_path):
            print(f"Loading model weights from: {model_path}")
            self.model.load_state_dict(torch.load(model_path, map_location=self.device), strict=False)
            self.model.eval()
        else:
            raise ValueError(f"Model file not found: {model_path}")   
        
        dummy_input = self.get_dummy_input()
        if isinstance(dummy_input, torch.Tensor):
            print(f"Dummy input shape: {dummy_input.shape}")
        elif isinstance(dummy_input, (list, tuple)):
            print(f"Dummy input shapes: {[x.shape for x in dummy_input if isinstance(x, torch.Tensor)]}")

        model_to_export = self.model.module if isinstance(self.model, torch.nn.DataParallel) else self.model
        
        # Export to ONNX
        torch.onnx.export(
            model_to_export,
            dummy_input,
            self.onnx_path,
            export_params=True,
            opset_version=self.onnx_config["opset_version"],
            do_constant_folding=True,
            input_names=self.onnx_config["input_names"],
            output_names= self.onnx_config["output_names"],
            dynamic_axes=self.onnx_config["dynamic_axes"],
        )
        
        print(f'Model exported to ONNX: {self.onnx_path}')
        
        # Verify the ONNX model
        try:
            onnx_model = onnx.load(self.onnx_path)
            onnx.checker.check_model(onnx_model)
            print('ONNX model verified successfully')
        except Exception as e:
            print(f'ONNX model verification warning: {e}')
        
        return self.onnx_path


    def save_test_outputs(self, predictions, labels, config):
    
        output_dir = config.TEST.OUTPUT_SAVE_DIR
        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
        
        # Filename ID to be used in any output files that get saved
        if config.TOOLBOX_MODE == 'train_and_test':
            filename_id = self.model_file_name
        elif config.TOOLBOX_MODE == 'only_test':
            model_file_root = config.INFERENCE.MODEL_PATH.split("/")[-1].split(".pth")[0]
            filename_id = model_file_root + "_" + config.TEST.DATA.DATASET
        else:
            raise ValueError('Metrics.py evaluation only supports train_and_test and only_test!')
        output_path = os.path.join(output_dir, filename_id + '_outputs.pickle')

        data = dict()
        data['predictions'] = predictions
        data['labels'] = labels
        data['label_type'] = config.TEST.DATA.PREPROCESS.LABEL_TYPE
        data['fs'] = config.TEST.DATA.FS

        with open(output_path, 'wb') as handle: # save out frame dict pickle file
            pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)

        print('Saving outputs to:', output_path)

    def plot_losses_and_lrs(self, train_loss, valid_loss, lrs, config):

        output_dir = os.path.join(config.LOG.PATH, config.TRAIN.DATA.EXP_DATA_NAME, 'plots')
        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)

        # Filename ID to be used in plots that get saved
        if config.TOOLBOX_MODE == 'train_and_test':
            filename_id = self.model_file_name
        else:
            raise ValueError('Metrics.py evaluation only supports train_and_test and only_test!')
        
        # Create a single plot for training and validation losses
        plt.figure(figsize=(10, 6))
        epochs = range(0, len(train_loss))  # Integer values for x-axis
        plt.plot(epochs, train_loss, label='Training Loss')
        if len(valid_loss) > 0:
            plt.plot(epochs, valid_loss, label='Validation Loss')
        else:
            print("The list of validation losses is empty. The validation loss will not be plotted!")
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title(f'{filename_id} Losses')
        plt.legend()
        plt.xticks(epochs)

        # Set y-axis ticks with more granularity
        ax = plt.gca()
        ax.yaxis.set_major_locator(MaxNLocator(integer=False, prune='both'))

        loss_plot_filename = os.path.join(output_dir, filename_id + '_losses.pdf')
        plt.savefig(loss_plot_filename, dpi=300)
        plt.close()

        # Create a separate plot for learning rates
        plt.figure(figsize=(6, 4))
        scheduler_steps = range(0, len(lrs))
        plt.plot(scheduler_steps, lrs, label='Learning Rate')
        plt.xlabel('Scheduler Step')
        plt.ylabel('Learning Rate')
        plt.title(f'{filename_id} LR Schedule')
        plt.legend()

        # Set y-axis values in scientific notation
        ax = plt.gca()
        ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True, useOffset=False))
        ax.ticklabel_format(axis='y', style='sci', scilimits=(0,0))  # Force scientific notation

        lr_plot_filename = os.path.join(output_dir, filename_id + '_learning_rates.pdf')
        plt.savefig(lr_plot_filename, bbox_inches='tight', dpi=300)
        plt.close()

        print('Saving plots of losses and learning rates to:', output_dir)
