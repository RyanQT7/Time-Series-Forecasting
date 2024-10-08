# coding=utf-8
# Copyright 2021 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" PyTorch VisionTextDualEncoder model."""
import sys
sys.path.insert(0, '/root/autodl-tmp/ViTST/code')

from typing import Optional

import torch
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

from transformers.modeling_utils import PreTrainedModel
from transformers.utils import add_start_docstrings, add_start_docstrings_to_model_forward, logging, replace_return_docstrings
from transformers.models.auto.configuration_auto import AutoConfig
from transformers.models.auto.modeling_auto import AutoModel

from transformers.models.clip.modeling_clip import CLIPOutput, CLIPVisionConfig, CLIPVisionModel
from transformers.utils import ModelOutput

# my vit/swin
from models.vit.modeling_vit import ViTConfig, ViTModel 
# from models1.vit_subimage.modeling_vit import ViTConfig, ViTModel # sub-images
from models.swin.modeling_swin import SwinConfig, SwinModel

from .configuration_vision_text_dual_encoder import VisionTextDualEncoderForClassificationConfig

from dataclasses import dataclass
from typing import Optional, Tuple

logger = logging.get_logger(__name__)

_CONFIG_FOR_DOC = "VisionTextDualEncoderForClassificationConfig"

VISION_TEXT_DUAL_ENCODER_START_DOCSTRING = r"""
    This class can be used to initialize a vision-text dual encoder model with any pretrained vision autoencoding model
    as the vision encoder and any pretrained text model as the text encoder. The vision and text encoders are loaded
    via the [`~AutoModel.from_pretrained`] method. The projection layers are automatically added to the model and
    should be fine-tuned on a downstream task, like contrastive image-text modeling.
    In [LiT: Zero-Shot Transfer with Locked-image Text Tuning](https://arxiv.org/abs/2111.07991) it is shown how
    leveraging pre-trained (locked/frozen) image and text model for contrastive learning yields significant improvment
    on new zero-shot vision tasks such as image classification or retrieval.
    After such a Vision-Text-Dual-Encoder model has been trained/fine-tuned, it can be saved/loaded just like any other
    models1 (see the examples for more information).
    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)
    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.
    Parameters:
        config ([`VisionEncoderDecoderConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""


VISION_TEXT_DUAL_ENCODER_TEXT_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you provide
            it.
            Indices can be obtained using [`PreTrainedTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.
            [What are input IDs?](../glossary#input-ids)
        attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:
            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.
            [What are attention masks?](../glossary#attention-mask)
        position_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0,
            config.max_position_embeddings - 1]`.
            [What are position IDs?](../glossary#position-ids)
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
"""

VISION_TEXT_DUAL_ENCODER_VISION_INPUTS_DOCSTRING = r"""
    Args:
        pixel_values (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`):
            Pixel values. Padding will be ignored by default should you provide it. Pixel values can be obtained using
            [`CLIPFeatureExtractor`]. See [`CLIPFeatureExtractor.__call__`] for details.
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
"""

VISION_TEXT_DUAL_ENCODER_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you provide
            it.
            Indices can be obtained using [`CLIPTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.
            [What are input IDs?](../glossary#input-ids)
        attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:
            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.
            [What are attention masks?](../glossary#attention-mask)
        position_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0,
            config.max_position_embeddings - 1]`.
            [What are position IDs?](../glossary#position-ids)
        pixel_values (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`):
            Pixel values. Padding will be ignored by default should you provide it. Pixel values can be obtained using
            a feature extractor (e.g. if you use ViT as the encoder, you should use [`ViTFeatureExtractor`]). See
            [`ViTFeatureExtractor.__call__`] for details.
        return_loss (`bool`, *optional*):
            Whether or not to return the contrastive loss.
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
"""

@dataclass
class VisionTextClassifierOutput(ModelOutput):
    """
    Base class for outputs of sentence classification models1.

    Args:
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
            Classification (or regression if config.num_labels==1) loss.
        logits (`torch.FloatTensor` of shape `(batch_size, config.num_labels)`):
            Classification (or regression if config.num_labels==1) scores (before SoftMax).
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """

    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    text_embeds: torch.FloatTensor = None
    image_embeds: torch.FloatTensor = None
    text_model_output: torch.FloatTensor = None
    vision_model_output: torch.FloatTensor = None


@add_start_docstrings(VISION_TEXT_DUAL_ENCODER_START_DOCSTRING)
class VisionTextDualEncoderModelForClassification(PreTrainedModel):
    config_class = VisionTextDualEncoderForClassificationConfig
    base_model_prefix = "vision_text_dual_encoder"

    def __init__(
        self,
        config: Optional[VisionTextDualEncoderForClassificationConfig] = None,
        vision_model: Optional[PreTrainedModel] = None,
        text_model: Optional[PreTrainedModel] = None,
        prediction_head_dropout: float = None
    ):

        if config is None and (vision_model is None or text_model is None):
            raise ValueError("Either a configuration or an vision and a text model has to be provided")

        if config is None:
            config = VisionTextDualEncoderForClassificationConfig.from_vision_text_configs(vision_model.config, text_model.config)
        else:
            if not isinstance(config, self.config_class):
                raise ValueError(f"config: {config} has to be of type {self.config_class}")

        # initialize with config
        super().__init__(config)

        if vision_model is None:
            if isinstance(config.vision_config, CLIPVisionConfig):
                vision_model = CLIPVisionModel(config.vision_config)
            else:
                vision_model = AutoModel.from_config(config.vision_config)

        if text_model is None:
            text_model = AutoModel.from_config(config.text_config)
        
        self.vision_model = vision_model
        self.text_model = text_model
        
        # make sure that the individual model's config refers to the shared config
        # so that the updates to the config will be synced
        self.vision_model.config = self.config.vision_config
        self.text_model.config = self.config.text_config

        self.vision_embed_dim = config.vision_config.hidden_size
        self.text_embed_dim = config.text_config.hidden_size
        self.projection_dim = config.projection_dim
        self.num_classes = config.num_classes

        # Classifier head
        if prediction_head_dropout is not None:
            self.dropout = nn.Dropout(prediction_head_dropout)
        else:
            self.dropout = nn.Dropout(0.1)
        
        self.projection = nn.Linear(self.vision_embed_dim+self.text_embed_dim, self.projection_dim, bias=False)
        self.activation = nn.Tanh()
        self.classifier = nn.Linear(self.projection_dim, self.num_classes) if self.num_classes > 0 else nn.Identity()

        # Initialize weights and apply final processing
        # self.post_init()
        
    @add_start_docstrings_to_model_forward(VISION_TEXT_DUAL_ENCODER_TEXT_INPUTS_DOCSTRING)
    def get_text_features(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        token_type_ids=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        Returns:
            text_features (`torch.FloatTensor` of shape `(batch_size, output_dim`): The text embeddings obtained by
            applying the projection layer to the pooled output of [`CLIPTextModel`].
        Examples:
        ```python
        >>> from transformers import VisionTextDualEncoderModel, AutoTokenizer
        >>> model = VisionTextDualEncoderModel.from_pretrained("clip-italian/clip-italian")
        >>> tokenizer = AutoTokenizer.from_pretrained("clip-italian/clip-italian")
        >>> inputs = tokenizer(["una foto di un gatto", "una foto di un cane"], padding=True, return_tensors="pt")
        >>> text_features = model.get_text_features(**inputs)
        ```"""
        text_outputs = self.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        pooled_output = text_outputs[1]

        return pooled_output

    @add_start_docstrings_to_model_forward(VISION_TEXT_DUAL_ENCODER_VISION_INPUTS_DOCSTRING)
    def get_image_features(
        self,
        pixel_values=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        Returns:
            image_features (`torch.FloatTensor` of shape `(batch_size, output_dim`): The image embeddings obtained by
            applying the projection layer to the pooled output of [`CLIPVisionModel`].
        Examples:
        ```python
        >>> from PIL import Image
        >>> import requests
        >>> from transformers import VisionTextDualEncoderModel, AutoFeatureExtractor
        >>> model = VisionTextDualEncoderModel.from_pretrained("clip-italian/clip-italian")
        >>> feature_extractor = AutoFeatureExtractor.from_pretrained("google/vit-base-patch16-224")
        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)
        >>> inputs = feature_extractor(images=image, return_tensors="pt")
        >>> image_features = model.get_image_features(**inputs)
        ```"""
        vision_outputs = self.vision_model(
            pixel_values=pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        pooled_output = vision_outputs[1]  # pooled_output

        return pooled_output

    @add_start_docstrings_to_model_forward(VISION_TEXT_DUAL_ENCODER_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=VisionTextClassifierOutput, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids=None,
        pixel_values=None,
        attention_mask=None,
        position_ids=None,
        labels=None,
        token_type_ids=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        Returns:
        Examples:
        ```python
        >>> from PIL import Image
        >>> import requests
        >>> from transformers import (
        ...     VisionTextDualEncoderModel,
        ...     VisionTextDualEncoderProcessor,
        ...     ViTFeatureExtractor,
        ...     BertTokenizer,
        ... )
        >>> tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        >>> feature_extractor = ViTFeatureExtractor.from_pretrained("google/vit-base-patch16-224")
        >>> processor = VisionTextDualEncoderProcessor(feature_extractor, tokenizer)
        >>> model = VisionTextDualEncoderModel.from_vision_vision_pretrained(
        ...     "google/vit-base-patch16-224", "bert-base-uncased"
        ... )
        >>> # contrastive training
        >>> urls = [
        ...     "http://images.cocodataset.org/val2017/000000039769.jpg",
        ...     "https://farm3.staticflickr.com/2674/5850229113_4fe05d5265_z.jpg",
        ... ]
        >>> images = [Image.open(requests.get(url, stream=True).raw) for url in urls]
        >>> inputs = processor(
        ...     text=["a photo of a cat", "a photo of a dog"], images=images, return_tensors="pt", padding=True
        ... )
        >>> outputs = model(
        ...     input_ids=inputs.input_ids,
        ...     attention_mask=inputs.attention_mask,
        ...     pixel_values=inputs.pixel_values,
        ...     return_loss=True,
        ... )
        >>> loss, logits_per_image = outputs.loss, outputs.logits_per_image  # this is the image-text similarity score
        >>> # save and load from pretrained
        >>> model.save_pretrained("vit-bert")
        >>> model = VisionTextDualEncoderModel.from_pretrained("vit-bert")
        >>> # inference
        >>> outputs = model(**inputs)
        >>> logits_per_image = outputs.logits_per_image  # this is the image-text similarity score
        >>> probs = logits_per_image.softmax(dim=1)  # we can take the softmax to get the label probabilities
        ```"""
        return_dict = return_dict if return_dict is not None else self.config.return_dict

        vision_outputs = self.vision_model(
            pixel_values=pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        text_outputs = self.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        """Different models1 use different embeddings for classification.
        Swin and bert both use pooler_output (outputs[1]) for classification. 
        Vit and roberta both use outputs[0][:,0,:] for classification
        TODO: choose different embeddings for classification according to image and text models1.
        """
        if self.vision_model.config.model_type == "swin":
            image_embeds = vision_outputs[1]  
        elif self.vision_model.config.model_type == "vit":
            # image_embeds = vision_outputs[0][:,0,:]
            image_embeds = vision_outputs[1]  
        else:
            raise Exception("Only support vit and swin vision model right now.")  

        if self.text_model.config.model_type == "bert":
            text_embeds = text_outputs[1]  
        elif self.text_model.config.model_type == "roberta":
            text_embeds = text_outputs[0][:,0,:]
        else:
            raise Exception("Only support bert and roberta text model now.")          
        
        vl_embeds = torch.concat([image_embeds, text_embeds], dim=-1)
        pooled_output = self.dropout(vl_embeds)
        pooled_output = self.projection(vl_embeds)
        pooled_output = self.activation(pooled_output)
        pooled_output = self.dropout(pooled_output)

        # classification logits
        logits = self.classifier(pooled_output)

        # calculate classification loss
        loss = None
        if labels is not None:
            if self.num_classes == 1:
                self.config.problem_type = "regression"
            elif self.num_classes > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                self.config.problem_type = "single_label_classification"
            else:
                self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_classes == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_classes), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)

        if not return_dict:
            output = (logits)
            return ((loss,) + output) if loss is not None else output

        return VisionTextClassifierOutput(
            loss=loss,
            logits=logits
        )

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        # At the moment fast initialization is not supported
        # for composite models1
        kwargs["_fast_init"] = False
        return super().from_pretrained(*args, **kwargs)

    @classmethod
    def from_vision_text_pretrained(
        cls,
        vision_model_name_or_path: str = None,
        text_model_name_or_path: str = None,
        num_classes: int = None,
        *model_args,
        **kwargs,
    ) -> PreTrainedModel:
        """
        Params:
            vision_model_name_or_path (`str`, *optional*, defaults to `None`):
                Information necessary to initiate the vision model. Can be either:
                    - A string, the *model id* of a pretrained model hosted inside a model repo on huggingface.co.
                      Valid model ids can be located at the root-level, like `bert-base-uncased`, or namespaced under a
                      user or organization name, like `dbmdz/bert-base-german-cased`.
                    - A path to a *directory* containing model weights saved using
                      [`~FlaxPreTrainedModel.save_pretrained`], e.g., `./my_model_directory/`.
                    - A path or url to a *PyTorch checkpoint folder* (e.g, `./pt_model`). In this case, `from_pt`
                      should be set to `True` and a configuration object should be provided as `config` argument. This
                      loading path is slower than converting the PyTorch checkpoint in a Flax model using the provided
                      conversion scripts and loading the Flax model afterwards.
            text_model_name_or_path (`str`, *optional*):
                Information necessary to initiate the text model. Can be either:
                    - A string, the *model id* of a pretrained model hosted inside a model repo on huggingface.co.
                      Valid model ids can be located at the root-level, like `bert-base-uncased`, or namespaced under a
                      user or organization name, like `dbmdz/bert-base-german-cased`.
                    - A path to a *directory* containing model weights saved using
                      [`~FlaxPreTrainedModel.save_pretrained`], e.g., `./my_model_directory/`.
                    - A path or url to a *PyTorch checkpoint folder* (e.g, `./pt_model`). In this case, `from_pt`
                      should be set to `True` and a configuration object should be provided as `config` argument. This
                      loading path is slower than converting the PyTorch checkpoint in a Flax model using the provided
                      conversion scripts and loading the Flax model afterwards.
            model_args (remaining positional arguments, *optional*):
                All remaning positional arguments will be passed to the underlying model's `__init__` method.
            kwargs (remaining dictionary of keyword arguments, *optional*):
                Can be used to update the configuration object (after it being loaded) and initiate the model (e.g.,
                `output_attentions=True`).
                - To update the text configuration, use the prefix *text_* for each configuration parameter.
                - To update the vision configuration, use the prefix *vision_* for each configuration parameter.
                - To update the parent model configuration, do not use a prefix for each configuration parameter.
                Behaves differently depending on whether a `config` is provided or automatically loaded.
        Example:
        ```python
        >>> from transformers import VisionTextDualEncoderModel
        >>> # initialize a model from pretrained ViT and BERT models1. Note that the projection layers will be randomly initialized.
        >>> model = VisionTextDualEncoderModel.from_vision_vision_pretrained(
        ...     "google/vit-base-patch16-224", "bert-base-uncased"
        ... )
        >>> # saving model after fine-tuning
        >>> model.save_pretrained("./vit-bert")
        >>> # load fine-tuned model
        >>> model = VisionTextDualEncoderModel.from_pretrained("./vit-bert")
        ```"""
        """
        参数:
        vision_model_name_or_path(' str '， *可选 *，默认为
        ' None '):
        启动视觉模型所需的信息。可以是:
        -一个字符串，在huggingface.co上的模型仓库中托管的预训练模型的 * 模型id *。
        有效的模型id可以位于根级，如
        ' bert-base-uncase '，或者位于用户或组织名称下的命名空间，如
        ' dbmdz/bert-base-german-case '。
        一个路径到一个包含模型权重的目录
        [' ~ FlaxPreTrainedModel。Save_pretrained ']，例如
        ' ./my_model_directory/ '。
        - PyTorch检查点文件夹 * 的路径或url(例如，' ./pt_model ')。在本例中，是
        ' from_pt '
        应该设置为
        ' True '，并且应该提供一个配置对象作为
        ' config '
        参数。这
        加载路径比使用提供的转换脚本在flex模型中转换PyTorch检查点并随后加载flex模型要慢。
        Text_model_name_or_path(' str '， *可选 *):
        启动文本模型所需的信息。可以是:
        -一个字符串，在huggingface.co上的模型仓库中托管的预训练模型的 * 模型id *。
        有效的模型id可以位于根级别，如
        ' bert-base-uncase '，或者位于a
        用户或组织名，如
        ' dbmdz/bert-base-german-case '。
        一个路径到一个包含模型权重的目录
        [' ~ FlaxPreTrainedModel。Save_pretrained ']，例如
        ' ./my_model_directory/ '。
        - PyTorch检查点文件夹 * 的路径或url(例如，' ./pt_model ')。在本例中，是
        ' from_pt '
        应该设置为
        ' True '，并且应该提供一个配置对象作为
        ' config '
        参数。这种加载路径比使用提供的转换脚本在flex模型中转换PyTorch检查点并随后加载flex模型要慢。
        Model_args(剩余位置参数，*可选 *):
        所有剩余的位置参数将被传递给底层模型的
        ' __init__ '
        方法。
        Kwargs(剩余的关键字参数字典，*可选 *):
        可用于更新配置对象(在它被加载后)
        和初始化模型(例如，
        “output_attentions = True”)。
        —如果需要更新文本配置，请在每个配置参数前加上前缀“*text_ *”。
        —如果需要更新视觉配置，请在每个配置参数前加上前缀“*vision_ *”。
        —为了更新父模型配置，每个配置参数不使用前缀。
        行为不同取决于是否提供或自动加载
        ' config '。
        例子:
        ”“python
        从transformer导入VisionTextDualEncoderModel
        >> >  # 从预训练的ViT和BERT模型初始化模型1。注意投影层将被随机初始化。
        >> > model = VisionTextDualEncoderModel.from_vision_text_pretrained(
        …“谷歌 / 维特 - 基地 - patch16 - 224”, “bert - base - uncased”
        …)
        >> >  # 保存微调后的模型
        > > > model.save_pretrained(“。 / vit - bert”)
        >> >  # 加载微调模型
        >> > model = VisionTextDualEncoderModel.from_pretrained("./ vitt -bert")
        """
        kwargs_vision = {
            argument[len("vision_") :]: value for argument, value in kwargs.items() if argument.startswith("vision_")
        }

        kwargs_text = {
            argument[len("text_") :]: value for argument, value in kwargs.items() if argument.startswith("text_")
        }

        # remove vision, text kwargs from kwargs
        for key in kwargs_vision.keys():
            del kwargs["vision_" + key]
        for key in kwargs_text.keys():
            del kwargs["text_" + key]

        # Load and initialize the vision and text model
        vision_model = kwargs_vision.pop("model", None)
        if vision_model is None:
            if vision_model_name_or_path is None:
                raise ValueError(
                    "If `vision_model` is not defined as an argument, a `vision_model_name_or_path` has to be defined"
                )

            if "config" not in kwargs_vision:
                vision_config = AutoConfig.from_pretrained(vision_model_name_or_path)

            if vision_config.model_type == "clip":
                kwargs_vision["config"] = vision_config.vision_config
                vision_model = CLIPVisionModel.from_pretrained(vision_model_name_or_path, *model_args, **kwargs_vision)
                # TODO: Should we use the pre-trained projection as well ?
            elif vision_config.model_type == "vit":
                # our modified vit
                vision_model = ViTModel.from_pretrained(vision_model_name_or_path, *model_args, **kwargs_vision)
            elif vision_config.model_type == "swin":
                # our modified swin
                vision_model = SwinModel.from_pretrained(vision_model_name_or_path, *model_args, **kwargs_vision)
            else:
                kwargs_vision["config"] = vision_config
                vision_model = AutoModel.from_pretrained(vision_model_name_or_path, *model_args, **kwargs_vision)

        text_model = kwargs_text.pop("model", None)
        if text_model is None:
            if text_model_name_or_path is None:
                raise ValueError(
                    "If `text_model` is not defined as an argument, a `text_model_name_or_path` has to be defined"
                )

            if "config" not in kwargs_text:
                text_config = AutoConfig.from_pretrained(text_model_name_or_path)
                kwargs_text["config"] = text_config

            text_model = AutoModel.from_pretrained(text_model_name_or_path, *model_args, **kwargs_text)

        # instantiate config with corresponding kwargs
        config = VisionTextDualEncoderForClassificationConfig.from_vision_text_configs(vision_model.config, text_model.config, num_classes, **kwargs)

        # init model
        model = cls(config=config, vision_model=vision_model, text_model=text_model)

        # the projection layers are always newly initialized when loading the model
        # using pre-trained vision and text model.
        logger.warning(
            "The projection layer and logit scale weights `['visual_projection.weight', 'text_projection.weight',"
            " 'logit_scale']` are newly initialized. You should probably TRAIN this model on a down-stream task to be"
            " able to use it for predictions and inference."
        )

        return model