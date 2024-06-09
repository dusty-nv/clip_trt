#!/usr/bin/env python3
import os
import json
import logging


def load_prompts(prompts, concat=False):
    """
    Load prompts from a list of txt or json files
    (or if these are strings, just return the strings)
    """
    if prompts is None:
        return None
        
    if isinstance(prompts, str):
        prompts = [prompts]
        
    prompt_list = []
    
    for prompt in prompts:
        ext = os.path.splitext(prompt)[1]
        
        if ext == '.json':
            logging.info(f"loading prompts from {prompt}")
            with open(prompt) as file:
                json_prompts = json.load(file)
            for json_prompt in json_prompts:
                if isinstance(json_prompt, dict):
                    prompt_list.append(json_prompt['text'])
                elif isinstance(json_prompt, str):
                    prompt_list.append(json_prompt)
                else:
                    raise TypeError(f"{type(json_prompt)}")
        elif ext == '.txt':
            with open(prompt) as file:
                prompt_list.append(file.read().strip())
        else:
            prompt_list.append(prompt)
     
    if concat:
        if concat == True:
            concat = ' '
        prompt_list = concat.join(prompt_list)
               
    return prompt_list
    
