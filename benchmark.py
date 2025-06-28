#!/usr/bin/env python3
"""
Performance benchmarking script for chest X-ray AI models
"""

import time
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
import psutil
import GPUtil
import json
import os
from datetime import datetime
import argparse

class ModelBenchmark:
    """Benchmark model performance across different configurations"""
    
    def __init__(self, model_path=None):
        self.model_path = model_path
        self.model = None
        self.results = {}
        
        # System info
        self.system_info = self.get_system_info()
        
    def get_system_info(self):
        """Get system information"""
        info = {
            'cpu_count': psutil.cpu_count(),
            'memory_total': psutil.virtual_memory().total / (1024**3),  # GB
            'python_version': f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
            'tensorflow_version': tf.__version__,
            'keras_version': keras.__version__
        }
        
        # GPU information
        try:
            gpus = GPUtil.getGPUs()
            if gpus:
                gpu = gpus[0]
                info['gpu_name'] = gpu.name
                info['gpu_memory'] = gpu.memoryTotal
                info['gpu_driver'] = gpu.driver
            else:
                info['gpu_name'] = 'None'
        except:
            info['gpu_name'] = 'Unknown'
        
        return info
    
    def load_model(self, model_path=None):
        """Load model for benchmarking"""
        if model_path:
            self.model_path = model_path
        
        if not self.model_path:
            raise ValueError("Model path must be provided")
        
        print(f"Loading model from: {self.model_path}")
        self.model = keras.models.load_model(self.model_path)
        print(f"✅ Model loaded: {self.model.input_shape} -> {self.model.output_shape}")
        
        return self.model
    
    def create_synthetic_data(self, batch_sizes, num_samples=1000):
        """Create synthetic data for benchmarking"""
        input_shape = self.model.input_shape[1:]  # Remove batch dimension
        
        synthetic_data = {}
        for batch_size in batch_sizes:
            num_batches = num_samples // batch_size
            data = []
            
            for _ in range(num_batches):
                batch = np.random.random((batch_size,) + input_shape).astype(np.float32)
                data.append(batch)
            
            synthetic_data[batch_size] = data
        
        return synthetic_data
    
    def benchmark_inference_speed(self, batch_sizes=[1, 8, 16, 32, 64], num_samples=1000):
        """Benchmark inference speed across different batch sizes"""
        print("🚀 Benchmarking inference speed...")
        
        if not self.model:
            raise ValueError("Model must be loaded first")
        
        # Create synthetic data
        synthetic_data = self.create_synthetic_data(batch_sizes, num_samples)
        
        results = {}
        
        for batch_size in batch_sizes:
            print(f"  Testing batch size: {batch_size}")
            
            data_batches = synthetic_data[batch_size]
            times = []
            
            # Warmup
            for i in range(3):
                _ = self.model.predict(data_batches[0], verbose=0)
            
            # Actual benchmarking
            for batch in data_batches:
                start_time = time.time()
                _ = self.model.predict(batch, verbose=0)
                end_time = time.time()
                times.append(end_time - start_time)
            
            # Calculate statistics
            mean_time = np.mean(times)
            std_time = np.std(times)
            throughput = batch_size / mean_time  # samples per second
            
            results[batch_size] = {
                'mean_time': mean_time,
                'std_time': std_time,
                'throughput': throughput,
                'samples_processed': len(data_batches) * batch_size
            }
            
            print(f"    Mean time: {mean_time:.4f}s ± {std_time:.4f}s")
            print(f"    Throughput: {throughput:.2f} samples/sec")
        
        self.results['inference_speed'] = results
        return results
    
    def benchmark_memory_usage(self, batch_sizes=[1, 8, 16, 32, 64]):
        """Benchmark memory usage across different batch sizes"""
        print("💾 Benchmarking memory usage...")
        
        if not self.model:
            raise ValueError("Model must be loaded first")
        
        results = {}
        
        for batch_size in batch_sizes:
            print(f"  Testing batch size: {batch_size}")
            
            # Clear any existing memory
            tf.keras.backend.clear_session()
            
            # Reload model
            model = keras.models.load_model(self.model_path)
            
            # Create synthetic batch
            input_shape = model.input_shape[1:]
            batch = np.random.random((batch_size,) + input_shape).astype(np.float32)
            
            # Measure memory before
            process = psutil.Process()
            memory_before = process.memory_info().rss / (1024**2)  # MB
            
            # GPU memory before (if available)
            gpu_memory_before = 0
            try:
                gpus = GPUtil.getGPUs()
                if gpus:
                    gpu_memory_before = gpus[0].memoryUsed
            except:
                pass
            
            # Run inference
            _ = model.predict(batch, verbose=0)
            
            # Measure memory after
            memory_after = process.memory_info().rss / (1024**2)  # MB
            
            # GPU memory after (if available)
            gpu_memory_after = 0
            try:
                gpus = GPUtil.getGPUs()
                if gpus:
                    gpu_memory_after = gpus[0].memoryUsed
            except:
                pass
            
            results[batch_size] = {
                'cpu_memory_mb': memory_after - memory_before,
                'gpu_memory_mb': gpu_memory_after - gpu_memory_before,
                'total_cpu_memory_mb': memory_after,
                'total_gpu_memory_mb': gpu_memory_after
            }
            
            print(f"    CPU Memory: {memory_after - memory_before:.2f} MB")
            print(f"    GPU Memory: {gpu_memory_after - gpu_memory_before:.2f} MB")
        
        self.results['memory_usage'] = results
        return results
    
    def benchmark_model_size(self):
        """Benchmark model size and complexity"""
        print("📏 Benchmarking model size...")
        
        if not self.model:
            raise ValueError("Model must be loaded first")
        
        # Model file size
        file_size_mb = os.path.getsize(self.model_path) / (1024**2)
        
        # Model parameters
        total_params = self.model.count_params()
        trainable_params = sum([tf.keras.backend.count_params(w) for w in self.model.trainable_weights])
        non_trainable_params = total_params - trainable_params
        
        # Model layers
        num_layers = len(self.model.layers)
        
        # Model complexity (FLOPs estimation)
        flops = self.estimate_flops()
        
        results = {
            'file_size_mb': file_size_mb,
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'non_trainable_parameters': non_trainable_params,
            'num_layers': num_layers,
            'estimated_flops': flops
        }
        
        self.results['model_size'] = results
        
        print(f"  File size: {file_size_mb:.2f} MB")
        print(f"  Total parameters: {total_params:,}")
        print(f"  Trainable parameters: {trainable_params:,}")
        print(f"  Number of layers: {num_layers}")
        print(f"  Estimated FLOPs: {flops:,}")
        
        return results
    
    def estimate_flops(self):
        """Estimate FLOPs for the model"""
        # This is a simplified estimation
        # For more accurate FLOP counting, consider using tensorflow-model-optimization
        
        total_flops = 0
        input_shape = self.model.input_shape[1:]
        
        for layer in self.model.layers:
            if hasattr(layer, 'kernel_size') and hasattr(layer, 'filters'):
                # Convolutional layer
                if hasattr(layer, 'kernel_size') and hasattr(layer, 'filters'):
                    kernel_flops = np.prod(layer.kernel_size) * layer.input_shape[-1] * layer.filters
                    output_elements = np.prod(layer.output_shape[1:])
                    total_flops += kernel_flops * output_elements
            elif hasattr(layer, 'units'):
                # Dense layer
                input_size = layer.input_shape[-1] if layer.input_shape[-1] else np.prod(input_shape)
                total_flops += input_size * layer.units
        
        return total_flops
    
    def benchmark_accuracy_vs_speed_tradeoff(self, test_data=None, batch_sizes=[1, 8, 16, 32]):
        """Benchmark accuracy vs speed tradeoff"""
        print("⚖️ Benchmarking accuracy vs speed tradeoff...")
        
        if test_data is None:
            print("  No test data provided, skipping accuracy benchmark")
            return {}
        
        results = {}
        
        for batch_size in batch_sizes:
            print(f"  Testing batch size: {batch_size}")
            
            # Time inference
            start_time = time.time()
            predictions = self.model.predict(test_data['x'], batch_size=batch_size, verbose=0)
            inference_time = time.time() - start_time
            
            # Calculate accuracy metrics (simplified)
            if 'y' in test_data:
                # Binary classification metrics
                from sklearn.metrics import roc_auc_score
                try:
                    auc_scores = []
                    for i in range(predictions.shape[1]):
                        auc = roc_auc_score(test_data['y'][:, i], predictions[:, i])
                        auc_scores.append(auc)
                    mean_auc = np.mean(auc_scores)
                except:
                    mean_auc = 0.0
            else:
                mean_auc = 0.0
            
            throughput = len(test_data['x']) / inference_time
            
            results[batch_size] = {
                'inference_time': inference_time,
                'throughput': throughput,
                'mean_auc': mean_auc,
                'efficiency_score': mean_auc / inference_time  # AUC per second
            }
            
            print(f"    Inference time: {inference_time:.4f}s")
            print(f"    Throughput: {throughput:.2f} samples/sec")
            print(f"    Mean AUC: {mean_auc:.4f}")
        
        self.results['accuracy_vs_speed'] = results
        return results
    
    def benchmark_quantization_impact(self):
        """Benchmark impact of model quantization"""
        print("🔢 Benchmarking quantization impact...")
        
        try:
            # Create quantized model
            converter = tf.lite.TFLiteConverter.from_keras_model(self.model)
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            quantized_tflite_model = converter.convert()
            
            # Save quantized model temporarily
            quantized_path = 'temp_quantized_model.tflite'
            with open(quantized_path, 'wb') as f:
                f.write(quantized_tflite_model)
            
            # Load quantized model
            interpreter = tf.lite.Interpreter(model_path=quantized_path)
            interpreter.allocate_tensors()
            
            # Get input and output details
            input_details = interpreter.get_input_details()
            output_details = interpreter.get_output_details()
            
            # Create test data
            input_shape = input_details[0]['shape']
            test_input = np.random.random(input_shape).astype(np.float32)
            
            # Benchmark original model
            start_time = time.time()
            original_pred = self.model.predict(test_input, verbose=0)
            original_time = time.time() - start_time
            
            # Benchmark quantized model
            interpreter.set_tensor(input_details[0]['index'], test_input)
            start_time = time.time()
            interpreter.invoke()
            quantized_pred = interpreter.get_tensor(output_details[0]['index'])
            quantized_time = time.time() - start_time
            
            # Calculate differences
            pred_diff = np.mean(np.abs(original_pred - quantized_pred))
            speedup = original_time / quantized_time
            
            # File sizes
            original_size = os.path.getsize(self.model_path) / (1024**2)
            quantized_size = len(quantized_tflite_model) / (1024**2)
            size_reduction = (original_size - quantized_size) / original_size * 100
            
            results = {
                'original_inference_time': original_time,
                'quantized_inference_time': quantized_time,
                'speedup_factor': speedup,
                'prediction_difference': pred_diff,
                'original_size_mb': original_size,
                'quantized_size_mb': quantized_size,
                'size_reduction_percent': size_reduction
            }
            
            self.results['quantization'] = results
            
            print(f"  Speedup: {speedup:.2f}x")
            print(f"  Size reduction: {size_reduction:.1f}%")
            print(f"  Prediction difference: {pred_diff:.6f}")
            
            # Clean up
            os.remove(quantized_path)
            
            return results
            
        except Exception as e:
            print(f"  Quantization benchmark failed: {e}")
            return {}
    
    def run_comprehensive_benchmark(self, test_data=None):
        """Run comprehensive benchmark suite"""
        print("🏁 Starting comprehensive benchmark...")
        print(f"System: {self.system_info['cpu_count']} CPUs, {self.system_info['memory_total']:.1f}GB RAM, GPU: {self.system_info['gpu_name']}")
        print("="*60)
        
        start_time = time.time()
        
        # Load model if not already loaded
        if not self.model:
            self.load_model()
        
        # Run benchmarks
        self.benchmark_model_size()
        print()
        
        self.benchmark_inference_speed()
        print()
        
        self.benchmark_memory_usage()
        print()
        
        if test_data:
            self.benchmark_accuracy_vs_speed_tradeoff(test_data)
            print()
        
        self.benchmark_quantization_impact()
        print()
        
        total_time = time.time() - start_time
        
        # Add metadata
        self.results['metadata'] = {
            'benchmark_date': datetime.now().isoformat(),
            'total_benchmark_time': total_time,
            'model_path': self.model_path,
            'system_info': self.system_info
        }
        
        print("="*60)
        print(f"✅ Comprehensive benchmark completed in {total_time:.2f}s")
        
        return self.results
    
    def save_results(self, output_path='benchmark_results.json'):
        """Save benchmark results to file"""
        with open(output_path, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        print(f"📁 Results saved to: {output_path}")
    
    def generate_report(self, output_path='benchmark_report.md'):
        """Generate markdown report"""
        with open(output_path, 'w') as f:
            f.write("# Model Benchmark Report\n\n")
            f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # System Information
            f.write("## System Information\n\n")
            sys_info = self.results.get('metadata', {}).get('system_info', {})
            f.write(f"- **CPU:** {sys_info.get('cpu_count', 'Unknown')} cores\n")
            f.write(f"- **Memory:** {sys_info.get('memory_total', 'Unknown'):.1f} GB\n")
            f.write(f"- **GPU:** {sys_info.get('gpu_name', 'Unknown')}\n")
            f.write(f"- **TensorFlow:** {sys_info.get('tensorflow_version', 'Unknown')}\n\n")
            
            # Model Information
            if 'model_size' in self.results:
                f.write("## Model Information\n\n")
                model_info = self.results['model_size']
                f.write(f"- **File Size:** {model_info.get('file_size_mb', 0):.2f} MB\n")
                f.write(f"- **Parameters:** {model_info.get('total_parameters', 0):,}\n")
                f.write(f"- **Layers:** {model_info.get('num_layers', 0)}\n")
                f.write(f"- **Estimated FLOPs:** {model_info.get('estimated_flops', 0):,}\n\n")
            
            # Inference Speed
            if 'inference_speed' in self.results:
                f.write("## Inference Speed\n\n")
                f.write("| Batch Size | Mean Time (s) | Throughput (samples/s) |\n")
                f.write("|------------|---------------|------------------------|\n")
                
                for batch_size, metrics in self.results['inference_speed'].items():
                    f.write(f"| {batch_size} | {metrics['mean_time']:.4f} | {metrics['throughput']:.2f} |\n")
                f.write("\n")
            
            # Memory Usage
            if 'memory_usage' in self.results:
                f.write("## Memory Usage\n\n")
                f.write("| Batch Size | CPU Memory (MB) | GPU Memory (MB) |\n")
                f.write("|------------|-----------------|------------------|\n")
                
                for batch_size, metrics in self.results['memory_usage'].items():
                    f.write(f"| {batch_size} | {metrics['cpu_memory_mb']:.2f} | {metrics['gpu_memory_mb']:.2f} |\n")
                f.write("\n")
            
            # Quantization Impact
            if 'quantization' in self.results:
                f.write("## Quantization Impact\n\n")
                quant = self.results['quantization']
                f.write(f"- **Speedup:** {quant.get('speedup_factor', 0):.2f}x\n")
                f.write(f"- **Size Reduction:** {quant.get('size_reduction_percent', 0):.1f}%\n")
                f.write(f"- **Prediction Difference:** {quant.get('prediction_difference', 0):.6f}\n\n")
            
            # Recommendations
            f.write("## Recommendations\n\n")
            if 'inference_speed' in self.results:
                speed_results = self.results['inference_speed']
                best_batch = max(speed_results.keys(), key=lambda x: speed_results[x]['throughput'])
                f.write(f"- **Optimal Batch Size:** {best_batch} (highest throughput)\n")
            
            if 'quantization' in self.results and self.results['quantization']:
                speedup = self.results['quantization'].get('speedup_factor', 0)
                if speedup > 1.5:
                    f.write("- **Quantization:** Recommended (significant speedup with minimal accuracy loss)\n")
                else:
                    f.write("- **Quantization:** Not recommended (minimal speedup)\n")
        
        print(f"📊 Report saved to: {output_path}")

def main():
    """Main function for command-line usage"""
    parser = argparse.ArgumentParser(description='Benchmark chest X-ray AI model performance')
    parser.add_argument('--model', required=True, help='Path to trained model file')
    parser.add_argument('--test-data', help='Path to test data (optional)')
    parser.add_argument('--output-dir', default='benchmark_results', help='Output directory')
    parser.add_argument('--batch-sizes', nargs='+', type=int, default=[1, 8, 16, 32], 
                       help='Batch sizes to test')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize benchmark
    benchmark = ModelBenchmark(args.model)
    
    # Load test data if provided
    test_data = None
    if args.test_data:
        try:
            # Assume test data is in numpy format
            test_data = np.load(args.test_data, allow_pickle=True).item()
            print(f"✅ Test data loaded: {len(test_data['x'])} samples")
        except Exception as e:
            print(f"⚠️ Could not load test data: {e}")
    
    # Run benchmark
    results = benchmark.run_comprehensive_benchmark(test_data)
    
    # Save results
    results_path = os.path.join(args.output_dir, 'benchmark_results.json')
    benchmark.save_results(results_path)
    
    # Generate report
    report_path = os.path.join(args.output_dir, 'benchmark_report.md')
    benchmark.generate_report(report_path)
    
    print(f"\n🎉 Benchmark completed!")
    print(f"📁 Results: {results_path}")
    print(f"📊 Report: {report_path}")

if __name__ == "__main__":
    import sys
    main()