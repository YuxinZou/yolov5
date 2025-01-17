import numpy as np
from collections import deque
import os
import os.path as osp
from os.path import exists as file_exists
import copy
import torch
import torch.nn.functional as F

from .kalman_filter import KalmanFilter
from . import matching
from .basetrack import BaseTrack, TrackState

# from tracking_utils.deep.reid_model_factory import show_downloadeable_models, get_model_url, get_model_name
# from tracking_utils.deep.reid.torchreid.utils import FeatureExtractor
# from tracking_utils.deep.reid.torchreid.utils.tools import download_url
#
#
# class STrackWithFeat(BaseTrack):
#     shared_kalman = KalmanFilter()
#
#     def __init__(self, tlwh, score, temp_feat, buffer_size=30):
#         super(STrackWithFeat, self).__init__()
#         # wait activate
#         self._tlwh = np.asarray(tlwh, dtype=np.float)
#         self.kalman_filter = None
#         self.mean, self.covariance = None, None
#         self.is_activated = False
#
#         self.score = score
#         self.score_list = []
#         self.tracklet_len = 0
#
#         self.smooth_feat = None
#         self.update_features(temp_feat)
#         self.features = deque([], maxlen=buffer_size)
#         self.alpha = 0.9
#
#     def update_features(self, feat):
#         feat /= np.linalg.norm(feat)
#         self.curr_feat = feat
#         if self.smooth_feat is None:
#             self.smooth_feat = feat
#         else:
#             self.smooth_feat = self.alpha * self.smooth_feat + (1 - self.alpha) * feat
#         self.features.append(feat)
#         self.smooth_feat /= np.linalg.norm(self.smooth_feat)
#
#     def predict(self):
#         mean_state = self.mean.copy()
#         if self.state != TrackState.Tracked:
#             mean_state[7] = 0
#         self.mean, self.covariance = self.kalman_filter.predict(mean_state, self.covariance)
#
#     @staticmethod
#     def multi_predict(stracks):
#         if len(stracks) > 0:
#             multi_mean = np.asarray([st.mean.copy() for st in stracks])
#             multi_covariance = np.asarray([st.covariance for st in stracks])
#             for i, st in enumerate(stracks):
#                 if st.state != TrackState.Tracked:
#                     multi_mean[i][7] = 0
#             multi_mean, multi_covariance = STrackWithFeat.shared_kalman.multi_predict(multi_mean, multi_covariance)
#             for i, (mean, cov) in enumerate(zip(multi_mean, multi_covariance)):
#                 stracks[i].mean = mean
#                 stracks[i].covariance = cov
#
#     def activate(self, kalman_filter, frame_id):
#         """Start a new tracklet"""
#         self.kalman_filter = kalman_filter
#         self.track_id = self.next_id()
#         self.mean, self.covariance = self.kalman_filter.initiate(self.tlwh_to_xyah(self._tlwh))
#
#         self.tracklet_len = 0
#         self.state = TrackState.Tracked
#         if frame_id == 1:
#             self.is_activated = True
#         # self.is_activated = True
#         self.frame_id = frame_id
#         self.start_frame = frame_id
#         self.score_list.append(self.score)
#
#     def re_activate(self, new_track, frame_id, new_id=False):
#         self.mean, self.covariance = self.kalman_filter.update(
#             self.mean, self.covariance, self.tlwh_to_xyah(new_track.tlwh)
#         )
#         self.tracklet_len = 0
#         self.state = TrackState.Tracked
#         self.is_activated = True
#         self.frame_id = frame_id
#         if new_id:
#             self.track_id = self.next_id()
#         self.score = new_track.score
#         self.score_list.append(self.score)
#
#     def update(self, new_track, frame_id, update_feature=True):
#         """
#         Update a matched track
#         :type new_track: STrack
#         :type frame_id: int
#         :type update_feature: bool
#         :return:
#         """
#         self.frame_id = frame_id
#         self.tracklet_len += 1
#
#         new_tlwh = new_track.tlwh
#         self.mean, self.covariance = self.kalman_filter.update(
#             self.mean, self.covariance, self.tlwh_to_xyah(new_tlwh))
#         self.state = TrackState.Tracked
#         self.is_activated = True
#
#         self.score = new_track.score
#         self.score_list.append(self.score)
#         if update_feature:
#             self.update_features(new_track.curr_feat)
#
#     @property
#     # @jit(nopython=True)
#     def tlwh(self):
#         """Get current position in bounding box format `(top left x, top left y,
#                 width, height)`.
#         """
#         if self.mean is None:
#             return self._tlwh.copy()
#         ret = self.mean[:4].copy()
#         ret[2] *= ret[3]
#         ret[:2] -= ret[2:] / 2
#         return ret
#
#     @property
#     # @jit(nopython=True)
#     def tlbr(self):
#         """Convert bounding box to format `(min x, min y, max x, max y)`, i.e.,
#         `(top left, bottom right)`.
#         """
#         ret = self.tlwh.copy()
#         ret[2:] += ret[:2]
#         return ret
#
#     @staticmethod
#     # @jit(nopython=True)
#     def tlwh_to_xyah(tlwh):
#         """Convert bounding box to format `(center x, center y, aspect ratio,
#         height)`, where the aspect ratio is `width / height`.
#         """
#         ret = np.asarray(tlwh).copy()
#         ret[:2] += ret[2:] / 2
#         ret[2] /= ret[3]
#         return ret
#
#     def to_xyah(self):
#         return self.tlwh_to_xyah(self.tlwh)
#
#     @staticmethod
#     # @jit(nopython=True)
#     def tlbr_to_tlwh(tlbr):
#         ret = np.asarray(tlbr).copy()
#         ret[2:] -= ret[:2]
#         return ret
#
#     @staticmethod
#     # @jit(nopython=True)
#     def tlwh_to_tlbr(tlwh):
#         ret = np.asarray(tlwh).copy()
#         ret[2:] += ret[:2]
#         return ret
#
#     def __repr__(self):
#         return 'OT_{}_({}-{})'.format(self.track_id, self.start_frame, self.end_frame)
#
#
# class BYTETrackerWithFeat(object):
#     def __init__(self, args, frame_rate=30, device='cuda'):
#         self.tracked_stracks = []  # type: list[STrack]
#         self.lost_stracks = []  # type: list[STrack]
#         self.removed_stracks = []  # type: list[STrack]
#
#         self.frame_id = 0
#         self.args = args
#         # self.det_thresh = args.track_thresh
#         self.det_thresh = args.track_thresh + 0.1
#         self.buffer_size = int(frame_rate / 30.0 * args.track_buffer)
#         self.max_time_lost = self.buffer_size
#         self.kalman_filter = KalmanFilter()
#
#         # reid model
#         model_name = get_model_name(args.reid_weights)
#         model_url = get_model_url(args.reid_weights)
#         if not file_exists(args.reid_weights) and model_url is not None:
#             import gdown
#             gdown.download(model_url, str(args.reid_weights), quiet=False)
#         elif file_exists(args.reid_weights):
#             pass
#         elif model_url is None:
#             print('No URL associated to the chosen reid weights. Choose between:')
#             show_downloadeable_models()
#             exit()
#
#         self.extractor = FeatureExtractor(
#             # get rid of dataset information DeepSort model name
#             model_name=model_name,
#             model_path=args.reid_weights,
#             device=str(device)
#         )
#
#     def _get_features(self, bbox_xyxy, ori_img):
#         im_crops = []
#         for box in bbox_xyxy:
#             x1, y1, x2, y2 = list(map(int, box))
#             im = ori_img[y1:y2, x1:x2]
#             im_crops.append(im)
#         if im_crops:
#             features = self.extractor(im_crops)
#         else:
#             features = np.array([])
#         features = features.cpu().numpy()
#         features /= np.linalg.norm(features, axis=1, keepdims=True)
#         # print(features.shape)
#         return features
#
#     def update(self, output_results, ori_img):
#         self.frame_id += 1
#         activated_starcks = []
#         refind_stracks = []
#         lost_stracks = []
#         removed_stracks = []
#
#         if output_results.shape[1] == 5:
#             scores = output_results[:, 4]
#             bboxes = output_results[:, :4]
#         else:
#             output_results = output_results.cpu().numpy()
#             scores = output_results[:, 4] * output_results[:, 5]
#             bboxes = output_results[:, :4]  # x1y1x2y2
#
#         id_feature = self._get_features(bboxes, ori_img)
#         # img_h, img_w = img_info[0], img_info[1]
#         # scale = min(img_size[0] / float(img_h), img_size[1] / float(img_w))
#         # bboxes /= scale
#         # print(f'output_results : {output_results}')
#         remain_inds = scores > self.args.track_thresh
#         inds_low = scores > 0.1
#         inds_high = scores < self.args.track_thresh
#
#         inds_second = np.logical_and(inds_low, inds_high)
#         dets_second = bboxes[inds_second]
#         id_feature_second = id_feature[inds_second]
#         dets = bboxes[remain_inds]
#         id_feature = id_feature[remain_inds]
#         scores_keep = scores[remain_inds]
#         scores_second = scores[inds_second]
#
#         if len(dets) > 0:
#             '''Detections'''
#             detections = [STrackWithFeat(STrackWithFeat.tlbr_to_tlwh(tlbr), s, f, 30) for
#                           (tlbr, s, f) in zip(dets, scores_keep, id_feature)]
#         else:
#             detections = []
#         # print(f'detections : {detections}')
#         ''' Add newly detected tracklets to tracked_stracks'''
#         unconfirmed = []
#         tracked_stracks = []  # type: list[STrackWithFeat]
#         for track in self.tracked_stracks:
#             if not track.is_activated:
#                 unconfirmed.append(track)
#             else:
#                 tracked_stracks.append(track)
#
#         ''' Step 2: First association, with high score detection boxes'''
#         strack_pool = joint_stracks(tracked_stracks, self.lost_stracks)
#         # print(f'strack_pool: {strack_pool}')
#         # Predict the current location with KF
#         STrackWithFeat.multi_predict(strack_pool)
#         dists = matching.embedding_distance(strack_pool, detections)
#         dists = matching.fuse_motion(self.kalman_filter, dists, strack_pool, detections)
#         matches, u_track, u_detection = matching.linear_assignment(dists, thresh=self.args.match_thresh)
#         for itracked, idet in matches:
#             track = strack_pool[itracked]
#             det = detections[idet]
#             if track.state == TrackState.Tracked:
#                 track.update(detections[idet], self.frame_id)
#                 activated_starcks.append(track)
#             else:
#                 track.re_activate(det, self.frame_id, new_id=False)
#                 refind_stracks.append(track)
#
#         ''' Step 3: Second association, with low score detection boxes'''
#         # association the untrack to the low score detections
#         if len(dets_second) > 0:
#             '''Detections'''
#             detections_second = [STrackWithFeat(STrackWithFeat.tlbr_to_tlwh(tlbr), s, f, 30) for
#                                  (tlbr, s, f) in zip(dets_second, scores_second, id_feature_second)]
#         else:
#             detections_second = []
#         r_tracked_stracks = [strack_pool[i] for i in u_track if strack_pool[i].state == TrackState.Tracked]
#         dists = matching.iou_distance(r_tracked_stracks, detections_second)
#         matches, u_track, u_detection_second = matching.linear_assignment(dists, thresh=0.5)
#         for itracked, idet in matches:
#             track = r_tracked_stracks[itracked]
#             det = detections_second[idet]
#             if track.state == TrackState.Tracked:
#                 track.update(det, self.frame_id)
#                 activated_starcks.append(track)
#             else:
#                 track.re_activate(det, self.frame_id, new_id=False)
#                 refind_stracks.append(track)
#
#         for it in u_track:
#             track = r_tracked_stracks[it]
#             if not track.state == TrackState.Lost:
#                 track.mark_lost()
#                 lost_stracks.append(track)
#
#         '''Deal with unconfirmed tracks, usually tracks with only one beginning frame'''
#         detections = [detections[i] for i in u_detection]
#         # print('######second######')
#         # print(f'detections: {detections}')
#         dists = matching.iou_distance(unconfirmed, detections)
#         if not self.args.mot20:
#             dists = matching.fuse_score(dists, detections)
#         matches, u_unconfirmed, u_detection = matching.linear_assignment(dists, thresh=0.7)
#         # print(f'matches: {matches}')
#         # print(f'u_track: {u_track}')
#         # print(f'u_detection: {u_detection}')
#         for itracked, idet in matches:
#             unconfirmed[itracked].update(detections[idet], self.frame_id)
#             activated_starcks.append(unconfirmed[itracked])
#         for it in u_unconfirmed:
#             track = unconfirmed[it]
#             track.mark_removed()
#             removed_stracks.append(track)
#
#         """ Step 4: Init new stracks"""
#         for inew in u_detection:
#             track = detections[inew]
#             if track.score < self.det_thresh:
#                 continue
#             track.activate(self.kalman_filter, self.frame_id)
#             activated_starcks.append(track)
#         # print(f'activated_starcks: {activated_starcks}')
#         """ Step 5: Update state"""
#         for track in self.lost_stracks:
#             if self.frame_id - track.end_frame > self.max_time_lost:
#                 track.mark_removed()
#                 removed_stracks.append(track)
#
#         # print('Ramained match {} s'.format(t4-t3))
#
#         self.tracked_stracks = [t for t in self.tracked_stracks if t.state == TrackState.Tracked]
#         self.tracked_stracks = joint_stracks(self.tracked_stracks, activated_starcks)
#         self.tracked_stracks = joint_stracks(self.tracked_stracks, refind_stracks)
#         self.lost_stracks = sub_stracks(self.lost_stracks, self.tracked_stracks)
#         self.lost_stracks.extend(lost_stracks)
#         self.lost_stracks = sub_stracks(self.lost_stracks, self.removed_stracks)
#         self.removed_stracks.extend(removed_stracks)
#         self.tracked_stracks, self.lost_stracks = remove_duplicate_stracks(self.tracked_stracks, self.lost_stracks)
#         # get scores of lost tracks
#         output_stracks = [track for track in self.tracked_stracks if track.is_activated]
#
#         return output_stracks


class STrack(BaseTrack):
    shared_kalman = KalmanFilter()

    def __init__(self, tlwh, score):
        super(STrack, self).__init__()
        # wait activate
        self._tlwh = np.asarray(tlwh, dtype=np.float)
        self._det_tlwh = self._tlwh
        self.kalman_filter = None
        self.mean, self.covariance = None, None
        if score > 0.6:
            self.is_activated = True
        else:
            self.is_activated = False

        self.score = score
        self.tracklet_len = 0

    def predict(self):
        mean_state = self.mean.copy()
        if self.state != TrackState.Tracked:
            mean_state[7] = 0
        self.mean, self.covariance = self.kalman_filter.predict(mean_state, self.covariance)

    @staticmethod
    def multi_predict(stracks):
        if len(stracks) > 0:
            multi_mean = np.asarray([st.mean.copy() for st in stracks])
            multi_covariance = np.asarray([st.covariance for st in stracks])
            for i, st in enumerate(stracks):
                if st.state != TrackState.Tracked:
                    multi_mean[i][7] = 0
            multi_mean, multi_covariance = STrack.shared_kalman.multi_predict(multi_mean, multi_covariance)
            for i, (mean, cov) in enumerate(zip(multi_mean, multi_covariance)):
                stracks[i].mean = mean
                stracks[i].covariance = cov

    def activate(self, kalman_filter, frame_id):
        """Start a new tracklet"""
        self.kalman_filter = kalman_filter
        self.track_id = self.next_id()
        self.mean, self.covariance = self.kalman_filter.initiate(self.tlwh_to_xyah(self._tlwh))

        self.tracklet_len = 0
        self.state = TrackState.Tracked
        if frame_id == 1:
            self.is_activated = True
        # self.is_activated = True
        self.frame_id = frame_id
        self.start_frame = frame_id

    def re_activate(self, new_track, frame_id, new_id=False):
        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance, self.tlwh_to_xyah(new_track.tlwh)
        )
        self.tracklet_len = 0
        self.state = TrackState.Tracked
        self.is_activated = True
        self.frame_id = frame_id
        if new_id:
            self.track_id = self.next_id()
        self.score = new_track.score

    def update(self, new_track, frame_id):
        """
        Update a matched track
        :type new_track: STrack
        :type frame_id: int
        :type update_feature: bool
        :return:
        """
        self.frame_id = frame_id
        self.tracklet_len += 1

        new_tlwh = new_track.tlwh
        self._det_tlwh = new_tlwh
        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance, self.tlwh_to_xyah(new_tlwh))
        self.state = TrackState.Tracked
        self.is_activated = True

        self.score = new_track.score

    # 新增的属性，检测框
    @property
    def det_tlwh(self):
        return self._det_tlwh

    @property
    def det_xyxy(self):
        ret = self._det_tlwh.copy()
        ret[2:] += ret[:2]
        return ret

    # 新增的属性，检测框的接地点坐标
    @property
    def cxcy(self):
        ret = self._det_tlwh.copy()
        ret[0] += ret[2] / 2
        ret[1] += ret[3]
        return ret[:2]

    @property
    # @jit(nopython=True)
    def tlwh(self):
        """Get current position in bounding box format `(top left x, top left y,
                width, height)`.
        """
        if self.mean is None:
            return self._tlwh.copy()
        ret = self.mean[:4].copy()
        ret[2] *= ret[3]
        ret[:2] -= ret[2:] / 2
        return ret

    @property
    # @jit(nopython=True)
    def tlbr(self):
        """Convert bounding box to format `(min x, min y, max x, max y)`, i.e.,
        `(top left, bottom right)`.
        """
        ret = self.tlwh.copy()
        ret[2:] += ret[:2]
        return ret

    @staticmethod
    # @jit(nopython=True)
    def tlwh_to_xyah(tlwh):
        """Convert bounding box to format `(center x, center y, aspect ratio,
        height)`, where the aspect ratio is `width / height`.
        """
        ret = np.asarray(tlwh).copy()
        ret[:2] += ret[2:] / 2
        ret[2] /= ret[3]
        return ret

    def to_xyah(self):
        return self.tlwh_to_xyah(self.tlwh)

    @staticmethod
    # @jit(nopython=True)
    def tlbr_to_tlwh(tlbr):
        ret = np.asarray(tlbr).copy()
        ret[2:] -= ret[:2]
        return ret

    @staticmethod
    # @jit(nopython=True)
    def tlwh_to_tlbr(tlwh):
        ret = np.asarray(tlwh).copy()
        ret[2:] += ret[:2]
        return ret

    def __repr__(self):
        return 'OT_{}_({}-{})'.format(self.track_id, self.start_frame, self.end_frame)


class BYTETracker(object):
    def __init__(self, args, frame_rate=30):
        self.tracked_stracks = []  # type: list[STrack]
        self.lost_stracks = []  # type: list[STrack]
        self.removed_stracks = []  # type: list[STrack]

        self.frame_id = 0
        self.args = args
        # self.det_thresh = args.track_thresh
        self.det_thresh = args.track_thresh + 0.1
        self.buffer_size = int(frame_rate / 30.0 * args.track_buffer)
        self.max_time_lost = self.buffer_size
        self.kalman_filter = KalmanFilter()

    def update(self, output_results):  # , img_info, img_size):
        self.frame_id += 1
        activated_starcks = []
        refind_stracks = []
        lost_stracks = []
        removed_stracks = []

        if output_results.shape[1] == 5:
            scores = output_results[:, 4]
            bboxes = output_results[:, :4]
        else:
            output_results = output_results.cpu().numpy()
            scores = output_results[:, 4] * output_results[:, 5]
            bboxes = output_results[:, :4]  # x1y1x2y2
        # img_h, img_w = img_info[0], img_info[1]
        # scale = min(img_size[0] / float(img_h), img_size[1] / float(img_w))
        # bboxes /= scale
        # print(f'output_results : {output_results}')
        remain_inds = scores > self.args.track_thresh
        inds_low = scores > 0.1
        inds_high = scores < self.args.track_thresh

        inds_second = np.logical_and(inds_low, inds_high)
        dets_second = bboxes[inds_second]
        dets = bboxes[remain_inds]
        scores_keep = scores[remain_inds]
        scores_second = scores[inds_second]

        if len(dets) > 0:
            '''Detections'''
            detections = [STrack(STrack.tlbr_to_tlwh(tlbr), s) for
                          (tlbr, s) in zip(dets, scores_keep)]
        else:
            detections = []
        # print(f'detections : {detections}')
        ''' Add newly detected tracklets to tracked_stracks'''
        unconfirmed = []
        tracked_stracks = []  # type: list[STrack]
        for track in self.tracked_stracks:
            if not track.is_activated:
                unconfirmed.append(track)
            else:
                tracked_stracks.append(track)

        ''' Step 2: First association, with high score detection boxes'''
        strack_pool = joint_stracks(tracked_stracks, self.lost_stracks)
        # print(f'strack_pool: {strack_pool}')
        # Predict the current location with KF
        STrack.multi_predict(strack_pool)
        dists = matching.iou_distance(strack_pool, detections)
        if not self.args.mot20:
            dists = matching.fuse_score(dists, detections)
        matches, u_track, u_detection = matching.linear_assignment(dists, thresh=self.args.match_thresh)
        # print(f'matches: {matches}')
        # print(f'u_track: {u_track}')
        # print(f'u_detection: {u_detection}')
        for itracked, idet in matches:
            track = strack_pool[itracked]
            det = detections[idet]
            if track.state == TrackState.Tracked:
                track.update(detections[idet], self.frame_id)
                activated_starcks.append(track)
            else:
                # 说明匹配到了前几帧跟丢的对象
                # 就给该目标更新fram_id等信息，对象一旦被激活，即便跟丢is_activated也是True
                track.re_activate(det, self.frame_id, new_id=False)
                refind_stracks.append(track)

        ''' Step 3: Second association, with low score detection boxes'''
        # association the untrack to the low score detections
        if len(dets_second) > 0:
            '''Detections'''
            detections_second = [STrack(STrack.tlbr_to_tlwh(tlbr), s) for
                                 (tlbr, s) in zip(dets_second, scores_second)]
        else:
            detections_second = []
        r_tracked_stracks = [strack_pool[i] for i in u_track if strack_pool[i].state == TrackState.Tracked]
        dists = matching.iou_distance(r_tracked_stracks, detections_second)
        matches, u_track, u_detection_second = matching.linear_assignment(dists, thresh=0.5)
        for itracked, idet in matches:
            track = r_tracked_stracks[itracked]
            det = detections_second[idet]
            if track.state == TrackState.Tracked:
                track.update(det, self.frame_id)
                activated_starcks.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False)
                refind_stracks.append(track)

        for it in u_track:
            track = r_tracked_stracks[it]
            if not track.state == TrackState.Lost:
                track.mark_lost()
                lost_stracks.append(track)

        '''Deal with unconfirmed tracks, usually tracks with only one beginning frame'''
        detections = [detections[i] for i in u_detection]
        # print('######second######')
        # print(f'detections: {detections}')
        dists = matching.iou_distance(unconfirmed, detections)
        if not self.args.mot20:
            dists = matching.fuse_score(dists, detections)
        matches, u_unconfirmed, u_detection = matching.linear_assignment(dists, thresh=0.7)
        # print(f'matches: {matches}')
        # print(f'u_track: {u_track}')
        # print(f'u_detection: {u_detection}')
        for itracked, idet in matches:
            unconfirmed[itracked].update(detections[idet], self.frame_id)
            activated_starcks.append(unconfirmed[itracked])
        for it in u_unconfirmed:
            track = unconfirmed[it]
            track.mark_removed()
            removed_stracks.append(track)

        """ Step 4: Init new stracks"""
        for inew in u_detection:
            track = detections[inew]
            if track.score < self.det_thresh:
                continue
            track.activate(self.kalman_filter, self.frame_id)
            activated_starcks.append(track)
        # print(f'activated_starcks: {activated_starcks}')
        """ Step 5: Update state"""
        for track in self.lost_stracks:
            if self.frame_id - track.end_frame > self.max_time_lost:
                track.mark_removed()
                removed_stracks.append(track)

        # print('Ramained match {} s'.format(t4-t3))

        self.tracked_stracks = [t for t in self.tracked_stracks if t.state == TrackState.Tracked]
        self.tracked_stracks = joint_stracks(self.tracked_stracks, activated_starcks)
        self.tracked_stracks = joint_stracks(self.tracked_stracks, refind_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.tracked_stracks)
        self.lost_stracks.extend(lost_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.removed_stracks)
        self.removed_stracks.extend(removed_stracks)
        self.tracked_stracks, self.lost_stracks = remove_duplicate_stracks(self.tracked_stracks, self.lost_stracks)
        # get scores of lost tracks
        output_stracks = [track for track in self.tracked_stracks if track.is_activated]

        return output_stracks


def joint_stracks(tlista, tlistb):
    exists = {}
    res = []
    for t in tlista:
        exists[t.track_id] = 1
        res.append(t)
    for t in tlistb:
        tid = t.track_id
        if not exists.get(tid, 0):
            exists[tid] = 1
            res.append(t)
    return res


def sub_stracks(tlista, tlistb):
    stracks = {}
    for t in tlista:
        stracks[t.track_id] = t
    for t in tlistb:
        tid = t.track_id
        if stracks.get(tid, 0):
            del stracks[tid]
    return list(stracks.values())


def remove_duplicate_stracks(stracksa, stracksb):
    pdist = matching.iou_distance(stracksa, stracksb)
    pairs = np.where(pdist < 0.15)
    dupa, dupb = list(), list()
    for p, q in zip(*pairs):
        timep = stracksa[p].frame_id - stracksa[p].start_frame
        timeq = stracksb[q].frame_id - stracksb[q].start_frame
        if timep > timeq:
            dupb.append(q)
        else:
            dupa.append(p)
    resa = [t for i, t in enumerate(stracksa) if not i in dupa]
    resb = [t for i, t in enumerate(stracksb) if not i in dupb]
    return resa, resb
