# CONLL_2021_nonnative_speech_perception
This git contains all the code and informations needed to reproduce the resuts of the following paper:

## Getting the discrimination task data results
The stimuli (extracted wav files and source files) for the discrimination task are available under the name WorldVowels on the Perceptimatic dataset website, you can download them here : https://docs.cognitive-ml.fr/perceptimatic/Downloads/downloads.html#perceptimatic-dataset-files. Be careful, the extracted and source wav files are sampled at 16000Hz. The discrimination task results are also available on the website, but we need a specific format of them for our code to work, so they are available in the file `data/discrimination_results.csv` 

## Overlap score
#### Getting the assimilation task results
The stimuli used for the assimilation task are available in the same folder than the discrimination task stimuli. The participants results are available in `data/assimilation_results.csv`

#### Computing the overlap scores
Once you have the humans' assimilation file and the discrimination file, you can start creating a predictor file, by first computing the overlap score:

`python add_overlap_score.py $discrimination_file $assimilation_file $predictor_file`

## Getting models' delta values
#### Models' representations
##### MFCCs
To compute mfccs and save their representations, we use the source files of the World Vowels dataset. You can use the following command line:

`python compute_mfccs.py $folder_wav $folder_out mfccs`
##### DPGMM

We followed the instructions of https://github.com/geomphon/CogSci-2019-Unsupervised-speech-and-human-perception and we use their French and English models to transform the source wavs of the WorldVowels dataset.

##### Wav2vec 2.0
Clone and install fairseq: https://github.com/pytorch/fairseq/

Download the checkpoints of the universal model and the fine-tuned models from the voxpopuli github: https://github.com/facebookresearch/voxpopuli/

The universal model is the 'all 23 languages' 10k base model, and the fine tuned models are the English and French models fine tuned (ASR) models (from Base 10k).

Put the files in the extract_from_wav2vec directory in the fairseq git directory.

Modify the script `extract_wav2vec_layers.py` following the instructions in the comments.

Simply do (in the fairseq directory):

`python extract_wav2vec_layers.py` 

#### Computing delta values
To compute delta values, you need to use the script `compute_distances_from_representations.py`

`python compute_distances_from_representations.py $model $layer $distance $folder_out`

Where `$model` is either mfccs, dpgmm_english, dpgmm_french,  wav2vec_10k, wav2vec_10k_en, wav2vec_10k_fr. `$layer` is only applicable for wav2vec models, in our paper we used transf4. `$distance` is either kl (for dpgmm models) or cosine. `$folder_out` is where the file with delta values will be saved.

Once the delta values for all the models are computed, you can add them to the general predictor file created above, by doing:

`python add_delta_values.py $folder_results $file_in $file_out`

Where `folder_results` is the folder where all the delta values files are, `$file_in` is the predictor file created above, and `$file_out` is the final predictor file.

## How to compare the predictors ?
A file with all predictor values is already available in the git (`data/predictor_file_complete.csv`), otherwise you can create it by using the previous steps.

### Log-likelihood
##### Simple
`python simple_probit_model.py $predictor_file $out_file $french $english`

With `$french` equals to True if you want to test the predictors on the French participant results, and same for `$english` for English participants
`$outfile` will contain the log likelihood values obtained from the predictors that are in the predictor file, one column per predictor.
##### Bootstrap
`python bootstrap_probit_model.py $predictor_file $out_file $nb_it $french $english`

Same than for the simple probit model, but with `nb_it` the number of iteration of bootstrapping you want to do.
You need to precise in the file how many cpu are available for the computation.
### Spearman correlation
##### Simple
`python simple_spearman_correlation.py $predictor_file $outfile`

Where `$outfile` will contain the spearman correlation obtained by all the predictor values in `predictr_file`


##### Bootstrap
`python bootstrap_spearman_correlation.py $predictor_file $nb_it $outfile`

Same than for the simple spearman correlation but using bootstraping, you need to precise the number of iteration needed by choosing `$nb_it`
You need to precise in the file how many cpu are available for the computation.

## Native effect evaluation
#### Simple
`python simple_native_effect.py $predictor_file $english_value $french_value $to_put_to_neg`

to_put_to_neg needs to be True for the overlap score.

#### Bootstrap

`python bootstrap_native_effect.py $predictor_file`
This only works with a complete file of predictors (containing the overlap, wav2vec delta values and dpgmm delta values)


