#define _CRT_SECURE_NO_WARNINGS
#include<stdio.h>
#include<opencv2/opencv.hpp>
#pragma comment(lib,"opencv_world320d.lib")

//#include "opencv/cv.h"
using namespace std;
using namespace cv;

void Msk_split(
	const Mat& img, //元画像
	const Mat& msk,
	Mat& img_dst  //結果画像
)
{

	// (1)画像を読み込む
	vector<Mat> img_b(1), img_g(1), img_r(1);
	vector<Mat> img_data(3);
	vector<Mat> img_end(3);
	Mat img_dst_b, img_dst_g, img_dst_r;
	Mat end_b, end_g, end_r;


	split(img, img_data);

	img_b[0] = img_data[0];
	img_g[0] = img_data[1];
	img_r[0] = img_data[2];


	merge(img_b, img_dst_b);
	merge(img_g, img_dst_g);
	merge(img_r, img_dst_r);


	bitwise_and(img_dst_b, msk, end_b);
	bitwise_and(img_dst_g, msk, end_g);
	bitwise_and(img_dst_r, msk, end_r);

	img_end[0] = end_b;
	img_end[1] = end_g;
	img_end[2] = end_r;


	merge(img_end, img_dst);
}
int main()
{

	//clock_t start = clock();

	VideoCapture video1("Snow02.mp4");

	if (!video1.isOpened()) {
		cout << "error" << endl;
		return -1;
	}
	Size frame1((int)video1.get(CAP_PROP_FRAME_WIDTH),
		(int)video1.get(CAP_PROP_FRAME_HEIGHT));


	Mat img_src;//原图像

	Mat img_src2;//背景图像
	Mat img_src3 = imread("background02.jpg");

	Mat msk, msk_inv, msk_th, msk_th_inv; //儅僗僋
	Mat mskn, mskn_inv;//愥埲奜偺巆偭偨儅僗僋
	Mat msk_car, msk_small;
	Mat msk_e, msk_d; //朿挘丂埑弅

	Mat dst;

	Mat img_bbw;
	Mat img_df;//差分图像
	Mat img_bw; //差分的灰度值变化后图像
	Mat msk_bw; //二值化后的灰度值图像

	//三帧图像进行对比，判定雪和车
	Mat bef, now, next;//±1的原图像保存用	
	Mat bef_diff, now_diff, next_diff; ////±1的差分图像保存用
	Mat bef_di, now_di, next_di; //膨胀缩小后的图像
	Mat bef_di_inv, next_di_inv, now_di_inv; //将mask图像反转
	Mat com_di, com_di_inv; //前と後を合わせたマスクとその反転

	Mat bef_mskn;

	Mat car;
	Mat temp;

	Mat now_car, bef_car;
	Mat now_hanten_wh, bef_hanten_wh;

	Mat end_msk;

	Mat roi_img;

	vector<Mat> img1
	{ imread("車9.png"),imread("車10.png"),imread("車11.png"),imread("車12.png"),imread("車13.png"),
		imread("車14.png"),imread("車15.png"),imread("車16.png"),imread("車17.png"),imread("車18.png"),imread("車19.png"),
		imread("車20.png"),imread("車21.png"),imread("車22.png"),imread("車23.png"),imread("車24.png"),imread("車25.png"),
		imread("車26.png"),imread("車27.png"),imread("車28.png"),imread("車29.png"),imread("車30.png"),imread("車31.png"),
		imread("車32.png"),imread("車33.png"),imread("車34.png"),imread("車35.png"),imread("車36.png"),imread("車37.png"),
		imread("車38.png"),imread("車39.png"),imread("車40.png"),imread("車41.png"),imread("車42.png"),imread("車43.png"),
		imread("車44.png"),imread("車45.png"),imread("車46.png"),
		
	};


	vector<Mat> img2
	{ imread("黒9.png"),imread("黒10.png"),imread("黒11.png"),imread("黒12.png"),imread("黒13.png"),
		imread("黒14.png"),imread("黒15.png"),imread("黒16.png"),imread("黒17.png"),imread("黒18.png"),imread("黒19.png"),
		imread("黒20.png"),imread("黒21.png"),imread("黒22.png"),imread("黒23.png"),imread("黒24.png"),imread("黒25.png"),
		imread("黒26.png"),imread("黒27.png"),imread("黒28.png"),imread("黒29.png"),imread("黒30.png"),imread("黒31.png"),
		imread("黒32.png"),imread("黒33.png"),imread("黒34.png"),imread("黒35.png"),imread("黒36.png"),imread("黒37.png"),
		imread("黒38.png"),imread("黒39.png"),imread("黒40.png"),imread("黒41.png"),imread("黒42.png"),imread("黒43.png"),
		imread("黒44.png"),imread("黒45.png"),imread("黒46.png"),
		
	};
	int count = 0; //2フレーム後から前とチェック
	Mat tmp_mskn;
	Mat img_dst; //切り取り画像



	Mat msk_bg;  //背景を切り取り
	Mat img_end; //結果画像


	Mat element8 = (Mat_<uchar>(3, 3) << 1, 1, 1, 1, 1, 1, 1, 1, 1);
	//Mat element4 = (Mat_<uchar>(3, 3) << 0, 1, 0, 1, 0, 1, 0, 1, 0);
	VideoWriter out("Result01.mp4", CV_FOURCC('D', 'I', 'V', 'X'), 24.0, Size(640, 360), true);

	Mat resized_img_src;
	Mat resized_carRange;
	Mat new_blackWindow;
	while (1) {

		video1 >> img_src;
		if (img_src.empty()) break;

		//imshow("元图像", img_src);
		Mat small_img_src;

		resize(img_src, small_img_src, cv::Size(), 0.5, 0.5);
		imshow("元画像", small_img_src);
		img_src2 = img_src3.clone();
		//imshow("背景画像", img_src2);
		absdiff(img_src, img_src2, img_df);
		//imshow("差分画像", img_df);
		cvtColor(img_df, img_bw, COLOR_BGR2GRAY);
		



		//------------画像の保存

		//処理フレームと±フレームの画像保存ｒ
		////差分画像3フレーム
		//if (count > 1) bef_diff = now_diff.clone();  //3フレームまで入れるものがない
		//if (count > 0) now_diff = next_diff.clone();//2フレームまで入れるものがない
		//next_diff = img_bw.clone(); //1フレーム目の映像　リアルタイムの画像

		//元画像3フレーム
		if (count > 1) bef = now.clone();  //3フレームまで入れるものがない
		if (count > 0) now = next.clone();//2フレームまで入れるものがない
		next = img_src.clone();  //1フレーム目の映像　リアルタイムの画像

		threshold(img_bw, msk_bw, 25, 255, THRESH_BINARY);//2値化
		//imshow("二值化画像", msk_bw);
		msk_d = msk_bw.clone();//縮小拡大して大きい雪と車だけ残す

		for (int i = 0; i < 2; i++) {
			erode(msk_d, msk_d, element8, Point(-1, -1), 1);
		}
		//imshow("収縮", msk_d);
		for (int j = 0; j < 16; j++) {
			dilate(msk_d, msk_d, element8, Point(-1, -1), 1);
		}
		//imshow("膨張", msk_d);
		////	out << msk_d;
			//--------------------------前後のフレーム画像を確保----------------------------------//

		if (count > 1) bef_di = now_di.clone();  //3フレームまで入れるものがない

		if (count > 0) now_di = next_di.clone();//2フレームまで入れるものがない
		next_di = msk_d.clone();  //1フレーム目の映像　リアルタイムの画像

		Mat bef_mskn;
		Mat tmp_mskn, btmp_mskn;
		Mat img_bw, mskn;
		//---------メモ---------
		 //2フレーム前  -＞  前画像（比較画像）
		 //1フレーム前  -＞　基準の画像（チェック画像）
		 //今のフレーム -＞　後画像（比較画像）


		if (count > 2) {

			bitwise_not(bef_di, bef_di_inv);
			bitwise_not(next_di, next_di_inv);
			bitwise_and(bef_di_inv, next_di_inv, com_di_inv);
			bitwise_not(com_di_inv, com_di);

			//bitwise_not(now_bw, now_bw_inv);
			bitwise_and(now_di, com_di, mskn);
			//imshow("大きい雪粒の除去", mskn);

			Mat color_car;
			Msk_split(now, mskn, color_car);
		//	imshow("車の色を還元する", color_car);

			Mat mapCC;

			Mat blackWindow = Mat::zeros(img_src.rows, img_src.cols, CV_8UC3);
			Mat whiteWindow = Mat::zeros(img_src.rows, img_src.cols, CV_8UC3);
			int cols = whiteWindow.cols;
			int rows = whiteWindow.rows;

			for (int j = 0; j < rows; j++) {
				for (int i = 0; i < cols; i++) {
					whiteWindow.at<cv::Vec3b>(j, i)[0] = 255;
					whiteWindow.at<cv::Vec3b>(j, i)[1] = 255;
					whiteWindow.at<cv::Vec3b>(j, i)[2] = 255;
				}
			}

			for (int num = 0; num < 37; num++) {
				//imshow("",img1[num]);

				//matchTemplate(color_car, img1[num], mapCC, TM_CCOEFF_NORMED);
				matchTemplate(img_src, img1[num], mapCC, TM_CCOEFF_NORMED);

				double maxCC;
				Point maxLoc;
				minMaxLoc(mapCC, NULL, &maxCC, NULL, &maxLoc);

				if (maxCC > 0.8)
				{
					Rect car_roi(maxLoc, (maxLoc + Point(img1[num].cols, img1[num].rows)));

					Mat carRange = img_src(car_roi);

					//imshow("マッチングした車の範囲", carRange);

					Mat car;
					carRange.copyTo(car, img2[num]);
					//imshow("車両部画像の背景部を透明にした画像", car);

					car.copyTo(blackWindow(car_roi));

					Mat hanten;
					bitwise_not(img2[num], hanten);
					hanten.copyTo(whiteWindow(car_roi));
					break;
				}
			}
			Mat carr, snowCar;
			carr = blackWindow.clone();

			blackWindow.copyTo(dst);
			img_src2.copyTo(dst, whiteWindow);

			snowCar = dst.clone();
			//imshow("whiteWindow", whiteWindow);
			//imshow("snowCar", snowCar);
			//imshow("carr", carr);

			if (waitKey(20) == 27)break;

			Mat img_bww, hanten_wh;
			cvtColor(carr, img_bww, COLOR_BGR2GRAY);
			threshold(img_bww, hanten_wh, 20, 255, THRESH_BINARY);
			//imshow("hanten_wh", hanten_wh);
			/*cvtColor(carr, img_bw, COLOR_BGR2GRAY);
			threshold(img_bw, mskn, 20, 255, THRESH_BINARY);*/


			if (count > 3)
				bef_car = now_car.clone();
			now_car = snowCar.clone();

			if (count > 3)
				bef_hanten_wh = now_hanten_wh.clone();
			now_hanten_wh = hanten_wh.clone();

			Mat img_now, img_bef;
			/*img_now = mskn.clone();
			tmp_mskn = mskn.clone();*/
			img_now = hanten_wh.clone();
			tmp_mskn = hanten_wh.clone();
			Mat HSV_h, HSV_or;

			img_bef = bef_hanten_wh.clone();
			btmp_mskn = bef_hanten_wh.clone();
			/*imshow("bef_car", bef_car);*/
			/*imshow("now_car", now_car);
			imshow("snowCar", snowCar);*/

			//--------車両上の雪除去---------
			if (count > 3) {
				//imshow("今のフレーム", now_car);
				//imshow("前のフレーム", bef_car);

			   //--------------------now frameの重心---------------------
				Point Max, Min;
				Point Max_big, Min_big;

				Mat mskn_Focus;
				mskn_Focus = now_car.clone();//重心
				vector<vector<Point> >contours;
				findContours(img_now, contours, CV_RETR_LIST, CV_CHAIN_APPROX_NONE);

				double max_size = 0;
				int max_id = -1;

				for (int i = 0; i < contours.size(); i++) { //contours.size()=>中身の要素の個数　車の個数
					if (contours[i].size() > max_size) {//contoursの個数が最も多いもの（複数選ばれる車のうちの1つ）を選ぶ
						max_size = contours[i].size();//要素の最大数を変更
						max_id = i;//要素の最大数の時の配列番号

					}
				}
				if (max_id != -1) {
					Moments mu = moments(contours[max_id]);//要素の数が最大数の時の配列を利用

					fillConvexPoly(tmp_mskn, contours[max_id], Scalar(255));

					mskn = tmp_mskn.clone();
					img_now = tmp_mskn.clone();

					Point2f mc = Point2f(mu.m10 / mu.m00, mu.m01 / mu.m00);//重心の座標
					//cout << mc << endl;
					circle(mskn_Focus, mc, 4, Scalar(100), 2, 4);//重心　表示

					Point2f m_max;
					Point2f m_min;

					int cx_min = mc.x, cy_min = mc.y;
					int cx_max = mc.x, cy_max = mc.y;

					for (int i = 0; max_id != -1 && i < contours[max_id].size(); i++) {
						if (contours[max_id][i].x < cx_min) cx_min = contours[max_id][i].x;
						if (contours[max_id][i].y < cy_min) cy_min = contours[max_id][i].y;
						if (contours[max_id][i].x > cx_max) cx_max = contours[max_id][i].x;
						if (contours[max_id][i].y > cy_max) cy_max = contours[max_id][i].y;
					}

					//車両の最大・最上のx・yの値
					Max_big.x = cx_max;
					Max_big.y = cy_max;
					Min_big.x = cx_min;
					Min_big.y = cy_min;

					if (max_id != -1) {
						rectangle(mskn_Focus, Max_big, Min_big, Scalar(0, 200, 0), 5, 8);//比較画像　囲い（確認）
					}

					//四角のサイズを半分に
					int x_size = cx_max - cx_min;
					int y_size = cy_max - cy_min;

					//車両の外接する四角形　その半分のサイズの四角形
					//x・yの最大・最小値
					Max.x = mc.x + x_size / 4;
					Max.y = mc.y + y_size / 4;
					Min.x = mc.x - x_size / 4;
					Min.y = mc.y - y_size / 4;
					rectangle(mskn_Focus, Min, Max, Scalar(255, 0, 0), 5, 8);//確認用
					//imshow("nowframeの重心",mskn_Focus);

					//--------------------before frameの重心---------------------
					Point bMax, bMin;
					Point bMax_big, bMin_big;

					Mat bmskn_Focus;
					bmskn_Focus = bef_car.clone();//重心
					vector<vector<Point> >bcontours;
					findContours(img_bef, bcontours, CV_RETR_LIST, CV_CHAIN_APPROX_NONE);

					double bmax_size = 0;
					int bmax_id = -1;

					for (int i = 0; i < bcontours.size(); i++) { //contours.size()=>中身の要素の個数　車の個数
						if (bcontours[i].size() > bmax_size) {//contoursの個数が最もも多いもの（複数選ばれる車のうちの1つ）を選ぶ
							bmax_size = bcontours[i].size();//要素の最大数を変更
							bmax_id = i;//要素の最大数の時の配列番号

						}
					}
					if (bmax_id != -1) {
						Moments bmu = moments(bcontours[bmax_id]);//要素の数が最大数の時の配列を利用

						fillConvexPoly(btmp_mskn, bcontours[bmax_id], Scalar(255));
						//imshow("tmp_mskn", tmp_mskn);

						//mskn = tmp_mskn.clone();
						img_bef = tmp_mskn.clone();

						Point2f bmc = Point2f(bmu.m10 / bmu.m00, bmu.m01 / bmu.m00);//重心の座標
						//cout << bmc << endl;
						circle(bmskn_Focus, bmc, 4, Scalar(100), 2, 4);//重心　表示

						Point2f bm_max;
						Point2f bm_min;

						int bcx_min = bmc.x, bcy_min = bmc.y;
						int bcx_max = bmc.x, bcy_max = bmc.y;

						for (int i = 0; bmax_id != -1 && i < bcontours[bmax_id].size(); i++) {
							if (bcontours[bmax_id][i].x < bcx_min) bcx_min = bcontours[bmax_id][i].x;
							if (bcontours[bmax_id][i].y < bcy_min) bcy_min = bcontours[bmax_id][i].y;
							if (bcontours[bmax_id][i].x > bcx_max) bcx_max = bcontours[bmax_id][i].x;
							if (bcontours[bmax_id][i].y > bcy_max) bcy_max = bcontours[bmax_id][i].y;
						}

						//車両の最大・最上のx・yの値
						bMax_big.x = bcx_max;
						bMax_big.y = bcy_max;
						bMin_big.x = bcx_min;
						bMin_big.y = bcy_min;

						if (bmax_id != -1) {
							rectangle(bmskn_Focus, bMax_big, bMin_big, Scalar(0, 0, 255), 5, 8);//比較画像　囲い（確認）
						}

						//四角のサイズを半分に
						int bx_size = bcx_max - bcx_min;
						int by_size = bcy_max - bcy_min;

						//車両の外接する四角形　その半分のサイズの四角形
						//x・yの最大・最小値
						bMax.x = bmc.x + bx_size / 4;
						bMax.y = bmc.y + by_size / 4;
						bMin.x = bmc.x - bx_size / 4;
						bMin.y = bmc.y - by_size / 4;
						rectangle(bmskn_Focus, bMin, bMax, Scalar(0, 255, 0), 5, 8);//確認用
						//imshow("befframeの重心", bmskn_Focus);

						//差分を求める
						//now:基準フレーム　bef:一つ前のフレーム
						//mskn:基準フレームのマスク　 bef_mskn:一つ前のフレームのマスク
						//mc:基準フレームの重心　 bef_c:一つ前のフレームの重心

						//処理画像と比較画像の車の中心のずれを取得
						Point diff;
						diff.x = mc.x - bmc.x;
						diff.y = mc.y - bmc.y;
						cout << diff << endl;

						Mat M = (Mat_<double>(2, 3) << 1.0, 0.0, diff.x, 0.0, 1.0, diff.y);
						Mat bef_pd_bg;

						bef_pd_bg.rows = bef_car.rows;
						bef_pd_bg.cols = bef_car.cols;
						warpAffine(bef_car, bef_pd_bg, M, bef_pd_bg.size(), CV_INTER_LINEAR, cv::BORDER_TRANSPARENT);
						//imshow("bef_car_warpAffine", bef_pd_bg);
						//射影変換 平移

						Mat now_cl = now_car.clone();
						Mat bef_cl = bef_car.clone();
						Mat bef_cut, now_cut;



						//背景切り抜き　車両のみの画像
						Msk_split(now_cl, now_hanten_wh, now_cut);
						//imshow("nowframe車両のみの画像", now_cut);

						/*bef_mskn = bef_hanten_wh.clone();
						Msk_split(bef_cl, bef_mskn, bef_cut);
						imshow("befframe車両のみの画像", bef_cut);*/


						Msk_split(bef_cl, bef_hanten_wh, bef_cut);
						//imshow("befframe車両のみの画像", bef_cut);

						//平行移動
						Mat bef_pd;
						bef_pd.rows = bef_cut.rows;
						bef_pd.cols = bef_cut.cols;
						//warpAffine(bef_cut, bef_pd, M, bef_pd.size(), CV_INTER_LINEAR, cv::BORDER_TRANSPARENT);//比較画像　車両のみの範囲
						//平行移動時に現れる端の白の除去
						//白のままだと、HSVの処理で差として出てしまうため端の白色を黒色にする。
						warpAffine(bef_cut, bef_pd, M, bef_pd.size(), 1, 0, (0, 0, 0));//比較画像　車両のみの範囲
						//imshow("平行移動したbefframe車両のみの画像", bef_pd);

						//------HSV-----------//

						Mat car_hsv[2];
						cvtColor(now_cut, car_hsv[0], COLOR_BGR2HSV);//HSV化
						cvtColor(bef_pd, car_hsv[1], COLOR_BGR2HSV);//1フレーム前
						int h_d = 0, s_d = 0, v_d = 0;

						Mat msk_hsv(now_cut.rows, now_cut.cols, CV_8UC1);
						int seigen_s = 0;
						int seigen_v = 0;

						for (int y = 0; y < now_cut.rows; y++) {
							for (int x = 0; x < now_cut.cols; x++) {

								//基準画像と比較画像のHSV　それぞれの差（絶対値）
								h_d = abs(car_hsv[0].data[y*car_hsv[0].step + (x*car_hsv[0].channels())] - car_hsv[1].data[y*car_hsv[1].step + (x*car_hsv[1].channels())]);
								s_d = abs(car_hsv[0].data[y * car_hsv[0].step + x * car_hsv[0].elemSize() + 1]- car_hsv[1].data[y * car_hsv[1].step + x * car_hsv[1].elemSize() + 1]) = 0;
								v_d = abs(car_hsv[0].data[y * car_hsv[0].step + x * car_hsv[0].elemSize() + 2] - car_hsv[1].data[y * car_hsv[1].step + x * car_hsv[1].elemSize() + 2]) = 0;

								//彩度と明度の差の±
								//比較画像-基準画像
								seigen_s = car_hsv[1].data[y * car_hsv[1].step + x * car_hsv[1].elemSize() + 1] - car_hsv[0].data[y * car_hsv[0].step + x * car_hsv[0].elemSize() + 1];
								//基準画像-比較画像
								seigen_v = car_hsv[0].data[y * car_hsv[0].step + x * car_hsv[0].elemSize() + 2] - car_hsv[1].data[y * car_hsv[1].step + x * car_hsv[1].elemSize() + 2];

								if (s_d > 30) {//彩度の差の絶対値が30より大きい場合
									if (seigen_s >= 0)
										msk_hsv.data[y*car_hsv[0].cols + x] = 0;//彩度が基準画像の方が低い（雪の彩度は低い）
									else
										msk_hsv.data[y*car_hsv[0].cols + x] = 255;//それ以外は雪ではない
								}
								else {//彩度の差の絶対値が30より小さい場合
									msk_hsv.data[y*car_hsv[0].cols + x] = 255;//雪と判断しない　しかし
									if (v_d>10&& seigen_v>=0)//明度の差の値が10以上離れている＋明度が基準画像の方が高い
										msk_hsv.data[y*car_hsv[0].cols + x] = 0;
								}
							}
						}
						//imshow("車両部マスク", msk_hsv);
						//HSVを使用した動物体上の雪除去
						Mat HSV_cut, hsv_not;
						bitwise_not(msk_hsv, hsv_not);
						Msk_split(now_car, msk_hsv, HSV_cut); //雪の部分を黒くする。

						Mat small_HSV_cut;
						resize(HSV_cut, small_HSV_cut, cv::Size(), 0.5, 0.5);
						imshow("雪の部分が黒いフレーム画像（車両のみ）", small_HSV_cut);

						Msk_split(bef_pd_bg, hsv_not, HSV_h);//雪の補充用

						//imshow("現フレームの雪の部分を前フレームから持ってくる画像（補間画像）", HSV_h);

						bitwise_or(HSV_cut, HSV_h, HSV_or);//合成

						//cout << HSV_or.cols << " " << HSV_or.rows << "  " << HSV_or.channels() << std::endl;;

						//imshow("結果", HSV_or);
						Mat small_HSV_or;

						resize(HSV_or, small_HSV_or, cv::Size(), 0.5, 0.5);
						imshow("結果画像", small_HSV_or);

					}

				}
				out << HSV_or;
			}

		}
		if (waitKey(30) == 27)break;
		count++;
	}
	cvWaitKey();
	return 0;
}
