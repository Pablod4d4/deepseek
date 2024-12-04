# Copyright 2023 The OPRO Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""The utility functions for prompting GPT and Google Cloud models."""

import time
import requests
import json

def call_deepseek_local_single_prompt(
    prompt, 
    model="deepseek-r1:latest",
    max_decode_steps=20,
    temperature=0.8
):
    """调用本地部署的DeepSeek模型"""
    api_url = "http://localhost:11434/api/generate"
    payload = {
        "model": model,
        "prompt": prompt,
        "options": {
            "temperature": temperature,
            "max_tokens": max_decode_steps
        }
    }

    try:
        response = requests.post(
            api_url,
            data=json.dumps(payload),
            headers={"Content-Type": "application/json"},
            timeout=60
        )
        
        if response.status_code == 200:
            full_response = ""
            for line in response.iter_lines():
                if line:
                    chunk = json.loads(line.decode('utf-8'))
                    full_response += chunk.get("response", "")
                    if chunk.get("done", False):
                        break
            return full_response
        else:
            print(f"API error: {response.status_code}, retrying...")
            time.sleep(5)
            return call_deepseek_local_single_prompt(
                prompt, 
                max_decode_steps=max_decode_steps, 
                temperature=temperature
            )

    except (requests.exceptions.ConnectionError, 
           requests.exceptions.Timeout,
           requests.exceptions.RequestException) as e:
        retry_time = 10
        print(f"Connection error occurred: {e}. Retrying in {retry_time} seconds...")
        time.sleep(retry_time)
        return call_deepseek_local_single_prompt(
            prompt, 
            max_decode_steps=max_decode_steps, 
            temperature=temperature
        )

def call_deepseek_local(
    inputs, 
    model="deepseek-r1:latest",
    max_decode_steps=20,
    temperature=0.8
):
    """批量处理输入的调用接口"""
    if isinstance(inputs, str):
        inputs = [inputs]
    
    outputs = []
    for input_str in inputs:
        output = call_deepseek_local_single_prompt(
            input_str,
            model=model,
            max_decode_steps=max_decode_steps,
            temperature=temperature
        )
        outputs.append(output)
    return outputs