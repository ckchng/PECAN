
data_dir="/media/ckchng/1TBHDD/Dropbox/craters/PECAN_to_be_uploaded/data/"

epsilon=20
alpha=0.6

px_thres=5.0
ab_thres=0.1
deg_thres=10
lower_alpha=0.5
starting_id=0
step=3
img_h=1024
img_w=1024

num_cores=4

result_dir="result.txt"

testing_data_dir="${data_dir}/example_testing_data.csv"

python3 PECAN.py --data_dir "$data_dir" \
		   --result_dir "$result_dir"\
		   --testing_data_dir "$testing_data_dir"\
		   --starting_id "$starting_id"\
		   --step "$step" \
		   --px_thres "$px_thres" \
   		   --ab_thres "$ab_thres" \
   		   --deg_thres "$deg_thres" --epsilon "$epsilon" \
		   --lower_alpha "$lower_alpha" \
   		   --alpha "$alpha" \
   		   --num_cores "$num_cores" \
   		   --img_w "$img_w" \
      		   --img_h "$img_h" \
   		   
