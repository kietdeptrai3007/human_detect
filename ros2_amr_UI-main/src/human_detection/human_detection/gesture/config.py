from pathlib import Path

# ---------------------------------------------------------------------------
# Đường dẫn
# ---------------------------------------------------------------------------
PROJECT_DIR = Path(__file__).resolve().parent      # thư mục chứa code này
WORKSPACE_DIR = PROJECT_DIR.parent                 # thư mục cha: chứa Dataset/, runs/, weights/

# Thư mục gốc chứa video RoCoG. File chia ghi đường dẫn dạng
#   real/ground/Halt/S01_10m_ground_label5_start1234.mp4
# rocog_labels.resolve_video_path() sẽ tự thử nhiều cách ghép (xem hàm đó), nên chỉ cần
# trỏ DATASET_ROOT tới thư mục "Dataset" của bạn là đủ.
DATASET_ROOT = WORKSPACE_DIR / "Dataset"

# Thư mục chứa các file chia train/val/test tải từ GitHub RoCoG-v2
# (real_ground_train_tsonly.txt, real_ground_val_tsonly.txt, real_ground_test.txt, ...)
ANNOTATION_DIR = DATASET_ROOT / "real_ground" / "annotations"

# Nơi extract_keypoints.py ghi file .npz (mỗi video một file)
KEYPOINT_DIR = PROJECT_DIR / "keypoints"

# Nơi train_ddnet.py ghi checkpoint, log
CHECKPOINT_DIR = PROJECT_DIR / "checkpoints"

# ---------------------------------------------------------------------------
# Phát hiện người (YOLO) và cắt ảnh
# ---------------------------------------------------------------------------
# Trọng số YOLO (ultralytics): model người bạn đã finetune. Dùng đúng model này cho cả lúc
# trích dữ liệu lẫn lúc chạy trên robot (model COCO gốc như yolov8n.pt phải tải qua mạng).
YOLO_WEIGHTS = str(WORKSPACE_DIR / "runs" / "detect" / "runs" / "detect"
                   / "robot_person_detect" / "weights" / "best.pt")
YOLO_CONF = 0.4          # ngưỡng tin cậy của bbox người
YOLO_IMGSZ = 640
# Tracking (ByteTrack): YOLO chạy với ngưỡng thấp để ByteTrack có detection điểm thấp cho lượt ghép
# thứ hai; nhánh cử chỉ vẫn chỉ nhận người có conf >= YOLO_CONF.
TRACK_CONF = 0.1
TRACKER_CFG_DEFAULT = str(PROJECT_DIR.parents[1] / "config" / "bytetrack_person.yaml")  # khi chạy từ source
CROP_MARGIN = 0.25       # nới bbox thêm 25% mỗi chiều (tay duỗi ra ngoài bbox thân)
BBOX_SMOOTHING = 0.5     # EMA cho bbox: 1.0 = không làm mượt, nhỏ hơn = mượt hơn
MAX_MISSING_FRAMES = 5   # YOLO trượt tối đa bấy nhiêu frame liền thì vẫn dùng bbox cũ

# Tên model pose đang dùng (chỉ để GHI LẠI vào file .npz và checkpoint, giúp phát hiện
# trường hợp train bằng bản lite mà robot chạy bản full).
POSE_MODEL_NAME = "pose_landmarker_lite"

# ---------------------------------------------------------------------------
# Thời gian và cửa sổ
# ---------------------------------------------------------------------------
FPS_TARGET = 10          # mọi clip được resample về FPS này (gom 24 và 29 FPS về một)
T = 32                   # số frame DD-Net nhìn mỗi lần -> 32 / 10 = 3,2 giây. Phải chia hết cho 8.
TRIM_SEC = 0.3           # bỏ bấy nhiêu giây ở đầu và cuối clip khi cắt cửa sổ (nhãn nhiễu)
EVAL_STRIDE = 4          # bước trượt (frame) của cửa sổ khi đánh giá theo video
MIN_OK_RATIO = 0.5       # video có ít hơn 50% frame dùng được thì bị loại khỏi train/eval

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
DDNET_FILTERS = 16       # 16 = bản "lite" của bài báo; thử 32 hoặc 64 nếu underfit
DDNET_DROPOUT = 0.1      # SpatialDropout trong các khối Conv1D (giống bài báo)
DDNET_FC_DROPOUT = 0.5   # Dropout ở 2 lớp fully connected (giống bài báo)

# ---------------------------------------------------------------------------
# Train
# ---------------------------------------------------------------------------
SEED = 42
BATCH_SIZE = 64
EPOCHS = 300
LR = 1e-3
WEIGHT_DECAY = 1e-4
PATIENCE = 40            # early stopping: bao nhiêu epoch val không cải thiện thì dừng
SAMPLES_PER_CLIP = 8     # mỗi epoch lấy ngẫu nhiên bấy nhiêu cửa sổ từ mỗi clip train
LABEL_SMOOTHING = 0.1

# Augment (chỉ áp dụng cho tập train)
AUG_ROTATE_DEG = 10.0
AUG_SCALE = (0.9, 1.1)
AUG_SPEED = (0.8, 1.2)   # co giãn tốc độ cử chỉ
AUG_NOISE_STD = 0.02     # nhiễu Gauss thêm vào toạ độ (đơn vị "độ rộng vai chuẩn")
AUG_JOINT_DROP_PROB = 0.3  # xác suất một mẫu bị "che" một khớp trong một đoạn frame
USE_MIRROR = False       # lật ngang: TẮT mặc định vì cử chỉ RoCoG gắn với tay cụ thể

# ---------------------------------------------------------------------------
# Pretrain bằng dữ liệu JHMDB có sẵn trong repo DD-Net (tuỳ chọn)
# ---------------------------------------------------------------------------
# Repo https://github.com/fandulu/DD-Net có thư mục data/JHMDB/GT_train_1.pkl ...
# (15 khớp, 2D, đã gán tay). Thứ tự khớp JHMDB:
#   0 neck, 1 belly, 2 face, 3 R shoulder, 4 L shoulder, 5 R hip, 6 L hip,
#   7 R elbow, 8 L elbow, 9 R knee, 10 L knee, 11 R wrist, 12 L wrist,
#   13 R ankle, 14 L ankle
# Nếu skeleton_adapter.py của bạn xếp 15 khớp theo ĐÚNG thứ tự này thì để None.
# Nếu khác, khai báo danh sách: JHMDB_TO_ADAPTER[i] = chỉ số khớp JHMDB đặt vào vị trí i
# của adapter.
JHMDB_TO_ADAPTER = None
JHMDB_FPS = 30           # JHMDB quay ở 30 FPS, clip chỉ dài ~1 giây (15-40 frame)
# Chỉ số khớp trong bố cục của ADAPTER dùng để chuẩn hoá dữ liệu JHMDB giống adapter
# (gốc = trung điểm 2 vai, thước = độ dài thân / TORSO_TO_SHOULDER_RATIO).
# Lấy thẳng từ skeleton_adapter.py để không bao giờ lệch khi bạn sửa bố cục khớp ở đó.
from .skeleton_adapter import JOINT_NAMES, TORSO_TO_SHOULDER_RATIO  # noqa: E402

ADAPTER_R_SHOULDER = JOINT_NAMES.index("r_shoulder")
ADAPTER_L_SHOULDER = JOINT_NAMES.index("l_shoulder")
ADAPTER_R_HIP = JOINT_NAMES.index("r_hip")
ADAPTER_L_HIP = JOINT_NAMES.index("l_hip")