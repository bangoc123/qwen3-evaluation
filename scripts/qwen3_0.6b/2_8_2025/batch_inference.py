# -*- coding: utf-8 -*-
"""Complete batch processing script for 8x RTX A6000 with thinking mode enabled - PARALLEL VERSION"""

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from transformers import AutoModelForCausalLM, AutoTokenizer
import pandas as pd
from tqdm import tqdm
import os
import time
import gc
import multiprocessing

# Enable optimizations
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

def estimate_time(total_samples, num_gpus=1, thinking_mode=True):
    """Simple time estimation"""
    # Base speeds (samples per second)
    if thinking_mode:
        speed_per_gpu = 0.3  # Much slower with thinking
    else:
        speed_per_gpu = 2.5  # Normal speed
    
    total_speed = speed_per_gpu * num_gpus * 0.85  # 85% efficiency
    minutes = (total_samples / total_speed) / 60
    
    print(f"⏱️  Estimated time: {minutes:.1f} minutes ({minutes/60:.1f} hours)")
    print(f"   {total_samples:,} samples × {num_gpus} GPUs × {'thinking' if thinking_mode else 'normal'} mode")
    return minutes

def show_progress(done, total, start_time):
    """Show simple progress"""
    if done == 0:
        return
    
    elapsed = time.time() - start_time
    speed = done / elapsed
    remaining = (total - done) / speed / 60 if speed > 0 else 0
    
    print(f"📊 {done}/{total} ({100*done/total:.1f}%) - {remaining:.1f} min remaining")

class SimpleA6000Processor:
    def __init__(self, model_name="Qwen/Qwen3-0.6B", device_id=0):
        self.model_name = model_name
        self.device = f'cuda:{device_id}'
        self.device_id = device_id
        
    def load_model(self):
        """Load model on specific GPU"""
        print(f"Loading model on {self.device}...")
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True,
            use_fast=True
        )
        
        # Add pad token if missing
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            device_map=self.device,
            # torch_dtype=torch.float16,
            trust_remote_code=True,
            low_cpu_mem_usage=True
        )
        
        self.model.eval()
        print(f"✓ Model loaded on {self.device}")
    
    def answer_the_question(self, row):
        """Process single question with thinking mode (original format)"""
        system_prompt = """
        You are an expert in product information. Your job is to generate an answer to a question about a product based on the provided product information.

        Instructions:
            0. Identify the key elements in the user's question:
                - What is the name of the product? (eg : Oppo Reno8 T is not the same as Oppo Reno8 T 5G, realme 6 is not the same as realme 6i)
                - What specific issue or aspect is the question asking about?

            1.Regarding the information used to answer the question:
                - You are only allowed to use the provided product information to answer the question.
                - The provided information may include multiple products with similar or entirely different names. Identify and extract information only about the specific product mentioned in the question.
                - If the provided product information does not include details related to the user's question, do not generate information on your own. Instead, respond with: "Sorry, the information you requested is currently unavailable."

            2. Regarding the answer:
                - Keep the original wording of the extracted product information if possible
                - For every detail in the answer, include the source it was extracted from.
                - If the information requested in the question exists, provide a complete, detailed, and specific answer for the product mentioned in the question. Avoid giving general responses.
                - The answer should be user-friendly and easy to understand.
                - The answer must be in Vietnamese.
        """

        question = row["generated_question"]
        reference_str = row["reference_str"]

        user_prompt = (
            f"Generate an answer for the question: {question}\n"
            f"Based on the following product information:\n\n{reference_str}\n\n"
            f"Only provide the answer in Vietnamese. Do not add introduction or conclusion."
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]

        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True  # ✅ THINKING MODE ENABLED
        )

        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.device)

        # Generate response
        with torch.no_grad():
            # with torch.amp.autocast('cuda'):  # Updated autocast syntax
            generated_ids = self.model.generate(
                **model_inputs,
                max_new_tokens=32768,  # Keep original high token limit for thinking
                pad_token_id=self.tokenizer.eos_token_id,
                use_cache=True
            )

        output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()

        # Parse thinking content
        try:
            # Find the index of </think> token (151668)
            index = len(output_ids) - output_ids[::-1].index(151668)
        except ValueError:
            index = 0

        thinking_content = self.tokenizer.decode(output_ids[:index], skip_special_tokens=True).strip("\n")
        content = self.tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")

        return thinking_content, content
    
    def process_batch_with_thinking(self, questions, references, batch_size=32):
        """Process batch with thinking mode - smaller batch size due to longer sequences"""
        all_thinking = []
        all_answers = []
        
        print(f"Processing {len(questions)} questions in batches of {batch_size} (with thinking)")
        
        for i in tqdm(range(0, len(questions), batch_size), desc=f"GPU {self.device_id}"):
            end_idx = min(i + batch_size, len(questions))
            batch_questions = questions[i:end_idx]
            batch_references = references[i:end_idx]
            
            batch_thinking = []
            batch_answers = []
            
            # Process each question individually for thinking mode
            for q, r in zip(batch_questions, batch_references):
                try:
                    # Create temporary row
                    temp_row = {
                        "generated_question": q,
                        "reference_str": r
                    }
                    
                    thinking, answer = self.answer_the_question(temp_row)
                    batch_thinking.append(thinking)
                    print(answer)
                    batch_answers.append(answer)
                    
                except Exception as e:
                    print(f"Error processing question: {e}")
                    batch_thinking.append("ERROR")
                    batch_answers.append("ERROR")
            
            all_thinking.extend(batch_thinking)
            all_answers.extend(batch_answers)
            
            # Clear memory more frequently due to longer sequences
            if i % (batch_size * 2) == 0:
                torch.cuda.empty_cache()
                gc.collect()
        
        return all_thinking, all_answers

def process_single_gpu_with_thinking(gpu_id, data_chunk, output_file):
    """Process data chunk on single GPU with thinking mode"""
    try:
        processor = SimpleA6000Processor(device_id=gpu_id)
        processor.load_model()
        
        questions = data_chunk["generated_question"].tolist()
        references = data_chunk["reference_str"].tolist()
        
        # Smaller batch size for thinking mode (longer sequences)
        batch_size = 16 if gpu_id == 0 else 12  # Reduced due to thinking mode
        
        print(f"GPU {gpu_id}: Processing {len(questions)} samples with thinking mode")
        start_time = time.time()
        
        # Show progress every 20% of the way
        progress_checkpoints = [len(questions) // 5 * i for i in range(1, 6)]
        
        thinking_results, answer_results = processor.process_batch_with_thinking(
            questions, references, batch_size
        )
        
        # Show progress updates
        for checkpoint in progress_checkpoints:
            if len(answer_results) >= checkpoint:
                show_progress(len(answer_results), len(questions), start_time)
        
        elapsed = time.time() - start_time
        throughput = len(answer_results) / elapsed
        
        print(f"GPU {gpu_id}: {throughput:.1f} samples/second (with thinking)")
        
        # Add results to dataframe
        data_chunk[f"{processor.model_name}_thinking"] = thinking_results
        data_chunk[f"{processor.model_name}_answer"] = answer_results
        
        # Save chunk results
        chunk_file = f"{output_file}_gpu{gpu_id}.csv"
        data_chunk.to_csv(chunk_file, index=False)
        print(f"GPU {gpu_id}: Saved to {chunk_file}")
        
        return data_chunk
        
    except Exception as e:
        print(f"Error on GPU {gpu_id}: {e}")
        import traceback
        traceback.print_exc()
        return None

def process_gpu_worker(gpu_id, input_file, output_file):
    """Worker function for parallel GPU processing"""
    try:
        # Load data chunk
        data_chunk = pd.read_csv(input_file)
        
        # Process on this GPU
        result = process_single_gpu_with_thinking(gpu_id, data_chunk, output_file.replace('.csv', ''))
        
        if result is not None:
            print(f"✅ GPU {gpu_id} worker completed: {len(result)} samples")
        else:
            print(f"❌ GPU {gpu_id} worker failed")
            
    except Exception as e:
        print(f"❌ GPU {gpu_id} worker error: {e}")
        import traceback
        traceback.print_exc()

def main_multi_gpu_thinking():
    """Main function using multiple GPUs with thinking mode - PARALLEL VERSION"""
    
    print("🚀 8x RTX A6000 Ultra-Fast Processing (WITH THINKING MODE - PARALLEL)")
    print("====================================================================")
    
    # Check available GPUs
    if not torch.cuda.is_available():
        print("❌ CUDA not available")
        return
    
    num_gpus = torch.cuda.device_count()
    print(f"Found {num_gpus} GPUs")
    
    if num_gpus < 8:
        print(f"⚠️ Using {num_gpus} GPUs instead of 4")
        num_gpus = min(num_gpus, 8)
    
    # Load and prepare data
    try:
        df = pd.read_csv("hoanghamobile-question-answer.csv")
        print(f"Original dataset size: {len(df)}")
        
        # Process ALL data, not just from row 200
        df_to_process = df[["_id", "generated_question", "question_references", "answer", "reference_str"]]
        df_to_process = df_to_process.reset_index(drop=True)
        print(f"Loaded {len(df_to_process)} samples for processing")
        
        # Show time estimate
        estimate_time(len(df_to_process), num_gpus, thinking_mode=True)
        
    except FileNotFoundError:
        print("❌ CSV file not found. Please ensure 'hoanghamobile-question-answer.csv' exists")
        return
    
    # Split data across GPUs
    chunk_size = len(df_to_process) // num_gpus
    chunks = []
    total_assigned = 0
    
    for i in range(num_gpus):
        start_idx = i * chunk_size
        if i == num_gpus - 1:  # Last GPU gets remaining data
            end_idx = len(df_to_process)
        else:
            end_idx = (i + 1) * chunk_size
        
        chunk = df_to_process.iloc[start_idx:end_idx].copy()
        chunks.append(chunk)
        total_assigned += len(chunk)
        print(f"GPU {i}: {len(chunk)} samples (rows {start_idx}-{end_idx-1})")
    
    print(f"Total samples assigned: {total_assigned}/{len(df_to_process)}")
    
    # ========== PARALLEL PROCESSING ==========
    start_time = time.time()
    output_base = "output_4xa6000_thinking"
    
    print(f"\n🚀 Starting PARALLEL processing on {num_gpus} GPUs with THINKING MODE...")
    
    # Set start method for CUDA compatibility
    if multiprocessing.get_start_method(allow_none=True) != 'spawn':
        multiprocessing.set_start_method('spawn', force=True)
    
    # Create processes for each GPU
    processes = []
    results_files = []
    
    for gpu_id, chunk in enumerate(chunks):
        # Save chunk to temporary file for the process
        temp_file = f"temp_chunk_gpu{gpu_id}.csv"
        chunk.to_csv(temp_file, index=False)
        
        result_file = f"{output_base}_gpu{gpu_id}.csv"
        results_files.append(result_file)
        
        # Create process
        process = multiprocessing.Process(
            target=process_gpu_worker,
            args=(gpu_id, temp_file, result_file)
        )
        processes.append(process)
        process.start()
        print(f"✅ Started GPU {gpu_id} process (THINKING MODE ENABLED)")
    
    # Wait for all processes to complete
    print(f"\n⏳ Waiting for {len(processes)} GPUs to complete...")
    for i, process in enumerate(processes):
        process.join()
        print(f"✅ GPU {i} completed")
    
    # Clean up temp files
    for gpu_id in range(num_gpus):
        temp_file = f"temp_chunk_gpu{gpu_id}.csv"
        if os.path.exists(temp_file):
            os.remove(temp_file)
    
    # Combine all results
    all_results = []
    for result_file in results_files:
        if os.path.exists(result_file):
            result_df = pd.read_csv(result_file)
            all_results.append(result_df)
            print(f"📁 Loaded {len(result_df)} results from {result_file}")
    
    if all_results:
        final_df = pd.concat(all_results, ignore_index=True)
        final_output = f"{output_base}_combined.csv"
        final_df.to_csv(final_output, index=False)
        
        total_time = time.time() - start_time
        total_samples = len(final_df)
        final_throughput = total_samples / total_time
        
        print(f"\n🎉 PARALLEL PROCESSING COMPLETE!")
        print(f"📊 Results:")
        print(f"   Total samples processed: {total_samples}")
        print(f"   Original samples: {len(df_to_process)}")
        print(f"   Success rate: {100*total_samples/len(df_to_process):.1f}%")
        print(f"   Total time: {total_time:.2f} seconds ({total_time/60:.1f} minutes)")
        print(f"   Overall throughput: {final_throughput:.1f} samples/second")
        print(f"   Estimated speedup: ~{final_throughput/0.3:.1f}x (vs single GPU thinking)")
        print(f"💾 Final output: {final_output}")
        
        # Show sample of results
        print(f"\n📋 Sample Results (WITH THINKING):")
        if len(final_df) > 0:
            sample = final_df.iloc[0]
            print(f"Question: {sample['generated_question'][:100]}...")
            if 'Qwen/Qwen3-0.6B_thinking' in sample:
                print(f"Thinking: {sample['Qwen/Qwen3-0.6B_thinking'][:100]}...")
                print(f"Answer: {sample['Qwen/Qwen3-0.6B_answer'][:100]}...")
    else:
        print("❌ No results generated")

def benchmark_thinking_mode():
    """Benchmark thinking mode on single GPU"""
    print("🧪 Benchmark Thinking Mode on Single GPU")
    print("========================================")
    
    try:
        df = pd.read_csv("hoanghamobile-question-answer.csv")
        df_test = df.iloc[200:205]  # Just 5 samples for thinking mode benchmark
        df_test = df_test[["_id", "generated_question", "question_references", "answer", "reference_str"]]
        
        processor = SimpleA6000Processor(device_id=0)
        processor.load_model()
        
        questions = df_test["generated_question"].tolist()
        references = df_test["reference_str"].tolist()
        
        print(f"Testing thinking mode with {len(questions)} samples...")
        start_time = time.time()
        
        thinking_results, answer_results = processor.process_batch_with_thinking(
            questions, references, batch_size=1
        )
        
        elapsed = time.time() - start_time
        throughput = len(answer_results) / elapsed
        
        print(f"Thinking mode throughput: {throughput:.2f} samples/second")
        print(f"Average time per sample: {elapsed/len(answer_results):.2f} seconds")
        
        # Show sample results
        if thinking_results and answer_results:
            print(f"\n📋 Sample Result:")
            print(f"Question: {questions[0]}")
            print(f"Thinking: {thinking_results[0][:200]}...")
            print(f"Answer: {answer_results[0][:200]}...")
            
    except Exception as e:
        print(f"Benchmark failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", action="store_true", help="Run thinking mode benchmark")
    parser.add_argument("--single-gpu", action="store_true", help="Use single GPU with thinking")
    args = parser.parse_args()
    
    if args.benchmark:
        benchmark_thinking_mode()
    elif args.single_gpu:
        # Process on single GPU with thinking
        try:
            df = pd.read_csv("hoanghamobile-question-answer.csv")
            df_to_process = df[["_id", "generated_question", "question_references", "answer", "reference_str"]]
            print(f"Processing {len(df_to_process)} samples on single GPU with thinking mode")
            process_single_gpu_with_thinking(0, df_to_process, "output_8xa6000_thinking")
        except Exception as e:
            print(f"Single GPU processing failed: {e}")
            import traceback
            traceback.print_exc()
    else:
        main_multi_gpu_thinking()