/*
* Multiple Tracking.
* See COPYRIGHT file at the top of the source tree.
*
* This product includes software developed by the
* STARGUE Project (http://www.stargue.org/).
*
* This program is free software: you can redistribute it and/or modify
* it under the terms of the GNU General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* This program is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU General Public License for more details.
*
* You should have received a copy of the STARGUE License Statement and
* the GNU General Public License along with this program. If not,
* see <http://www.lsstcorp.org/LegalNotices/>.
*/

/**
 * @file DrawWithBlob.h
 *
 * @brief Blob related functions. 
 *
 * @author Adama Zouma
 * 
 * @Contact: stargue49@gmail.com
 *
 */

#ifndef DRAW_WITH_BLOB_H        
#define DRAW_WITH_BLOB_H

#include<opencv2/core/core.hpp>
#include<opencv2/highgui/highgui.hpp>
#include<opencv2/imgproc/imgproc.hpp>
#include<iostream>
#include "ConstantsTracking.h"
#include "Blob.h"

namespace za {
/* ============================================================================
 * Function Declaration
 * ============================================================================
 */

/**
 * \brief Predict mouse position.
 *
 * \details Rely on current mouse position to determine its next position.
 *
 * \param existingBlobs [in] mouse position, type is vector of opencv Point
 * \param currentFrameBlobs [in] mouse position, type is vector of opencv Point
 *
 * \return Next position, type is opencv Point
 *  
 */
double distanceBetweenPoints(cv::Point point1, cv::Point point2);

/**
 * \brief Predict mouse position.
 *
 * \details Rely on current mouse position to determine its next position.
 *
 * \param currentFrameBlob [in] mouse position, type is vector of opencv Point
 * \param existingBlobs [in] mouse position, type is vector of opencv Point
 * \param intIndex [in] index, type is int
 *
 * \return type is void
 *  
 */
void drawAndShowContours(cv::Size imageSize, std::vector<std::vector<cv::Point> > contours, std::string strImageName);

/**
 * \brief Add new blob.
 *
 * \details Add new blob.
 *
 * \param currentFrameBlob [in] current blob frame, type is vector of Blob
 * \param existingBlobs [in] existing blobs, type is vector of opencv Point
 *
 * \return type is void
 *  
 */
void drawAndShowContours(cv::Size imageSize, std::vector<za::Blob> blobs, std::string strImageName);

/**
 * \brief Add new blob.
 *
 * \details Add new blob.
 *
 * \param currentFrameBlob [in] current blob frame, type is vector of Blob
 * \param existingBlobs [in] existing blobs, type is vector of opencv Point
 *
 * \return type is void
 *  
 */
void drawBlobInfoOnImage(std::vector<za::Blob> &blobs, cv::Mat &imgFrame2Copy);

/**
 * \brief Draw car counter.
 *
 * \details Add new blob.
 *
 * \param carCount [in] car counter, type is int
 * \param imgFrame2Copy [in] frame, type is opencv Mat
 *
 * \return type is void
 *  
 */
void drawCarCountOnImage(int &carCount, cv::Mat &imgFrame2Copy);
}


# endif	// DRAW_WITH_BLOB_H
