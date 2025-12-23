# Config

## model_name
* bilora_ac_com_sdc_ema_auto: phiên bản thay đổi cách ghép weight W so với Moal gốc: thay vì expand W bằng ma trận zero thì concat các head của model (tương tự như concat head của BiLORA)
* adapt_ac_com_sdc_ema_auto: phiên bản gốc của Moal

## backbone_type
Nếu muốn pass task hiện tại thì sử dụng backbone: "vit_base_patch16_224_bilora_adapter"
