import H5AudioPlayer from "react-h5-audio-player";
import './AudioSucess.css'
import { DownloadIcon } from "../Icons/DownloadIcon.tsx";


export function AudioSuccess ({ className = '', src, ...rest }: typeof H5AudioPlayer.defaultProps) {
  return (
    <div className={`relative w-96 ${className}`}>
      <H5AudioPlayer
        className={`rounded-b-lg rounded-tr-lg bg-[var(--card-background-color)]`}
        style={{ boxShadow: 'none' }}
        autoPlay={true}
        src={src}
        {...rest}
      >
      </H5AudioPlayer>
      <a
        className="absolute top-[53%] left-[14%] text-[var(--text-faded-color)] hover:shadow-sm h-fit hover:opacity-75"
        href={src}
        download
        target="_blank"
      >
        <DownloadIcon className={'hover:font-bold'}/>
      </a>
    </div>
  )
}
