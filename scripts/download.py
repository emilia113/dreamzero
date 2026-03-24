from huggingface_hub import snapshot_download                                                             
                                                                                                            
snapshot_download(                                                                                        
    repo_id="GEAR-Dreams/DreamZero-DROID-Data",                                                           
    repo_type="dataset",                                                                                  
    local_dir="/datadrive/wjy/dataset/DreamZero-DROID-Data",
    max_workers=2,        # 限制并发为 2（默认是 8）                                                      
)