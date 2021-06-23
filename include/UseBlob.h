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
 * @file UseBlob.h
 *
 * @brief Blob related functions. 
 *
 * @author Adama Zouma
 * 
 * @Contact: stargue49@gmail.com
 *
 */
#ifndef USE_BLOB_H        
#define USE_BLOB_H

#include<opencv2/core/core.hpp>
#include<opencv2/highgui/highgui.hpp>
#include<opencv2/imgproc/imgproc.hpp>
#include<iostream>
#include "Blob.h"

namespace za {
/* ============================================================================
 * Function Declaration
 * ============================================================================
 */

/**
 * \brief Distance between two points.
 *
 * \details Distance between two points.
 *
 * \param point1 [in] first point, type is opencv Point
 * \param point2 [in] second point, type is opencv Point
 *
 * \return type is double
 *  
 */
double distanceBetweenPoints(cv::Point point1, cv::Point point2); 

/**
 * \brief Match current frame Blobs to existing Blobs.
 *
 * \details Match current frame Blobs to existing Blobs.
 *
 * \param existingBlobs [in] existing Blobs, type is vector of Blob
 * \param currentFrameBlobs [in] curent Blobs, type is vector of Blob
 *
 * \return type is void
 *  
 */
void matchCurrentFrameBlobsToExistingBlobs(std::vector<za::Blob> &existingBlobs, std::vector<za::Blob> &currentFrameBlobs);

/**
 * \brief Add Blob to existing Blobs.
 *
 * \details Add Blob to existing Blobs.
 *
 * \param currentFrameBlob [in] curent Blob, type is Blob
 * \param existingBlobs [in] [in] existing Blobs, type is vector of Blob
 * \param intIndex [in] index, type is int
 *
 * \return type is void
 *  
 */
void addBlobToExistingBlobs(za::Blob &currentFrameBlob, std::vector<za::Blob> &existingBlobs, int &intIndex);

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
void addNewBlob(za::Blob &currentFrameBlob, std::vector<za::Blob> &existingBlobs);

/**
 * \brief Add new blob.
 *
 * \details Add new blob.
 *
 * \param blobs [in] current blob frame, type is vector of Blob
 * \param intHorizontalLinePosition [in] horizontal line position, type is int
 * \param carCount [in] car counter, type is int
 *
 * \return type is void
 *  
 */
bool checkIfBlobsCrossedTheLine(std::vector<Blob> &blobs, int &intHorizontalLinePosition, int &carCount);
}

# endif	// USE_BLOB_H