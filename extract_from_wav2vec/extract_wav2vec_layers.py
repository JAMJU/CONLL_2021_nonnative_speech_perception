import wav2vec2_general_test
# to work the code needs to have the stimuli as wav files at 16000Hz with 1c
stimuli_path_wav = "" # replace this by the path to your stimuli to transform (extracted not source otherwise it will not work)

# WARNINGS : the wav files need to be mono and sampled at 16000Hz
def main(layer):
    writer = wav2vec2_general_test.EmbeddingDatasetWriter(
                input_root=stimuli_path_wav,
                output_root='PATHOUTPUT' + ''.join(layer.split('_')), # where you want to keep your extractions
                model_folder = 'FOLDERCHECKPOINT', # the folder where the checkpoint is (need to contain vocabulary file for the fine tuned models)
                model_fname='FNAMECHECKPOINT', # path to the checkpoint
                gpu=0, # do not change this
                extension='wav', # the extenson of your stimuli files
                use_feat=layer, # what kind of feature you want: 'conv_i' (i from 0 to 6) to extract from the convolutions of the encoder, 'z', 'q' or 'c' (following the naming of wav2vec 2.0 paper), or transf_0 to 12
                asr = True, # True if you are using a fine tuned model, False otherwise
            )

    print(writer)
    writer.require_output_path()

    print("Writing Features...")
    writer.write_features()
    print("Done.")

for lay in [ 'transf_4']: # change this list to the layers you want
    main(lay)
